#!/usr/bin/env python3
"""Build NB2 §7, "What is learned first" (Training Dynamics day).

Page shape (readings first, 2026-10-14): a short pitch and a one-line route, then one
`## Readings` section (#td27-readings) with every paper of the notebook, grouped by
section, then the derivations and walkthrough: 7.1 two moments (#td27-1), 7.2 higher
cumulants (#td27-2) and 7.3 spectral bias (#td27-3). Each opens with the question and
answer in plain words, a one-line paper pointer and a short core idea, then (from the
#td27-N-walkthrough anchor) the derivations, the exercises (bold labels, not headings)
with worked solutions, and the code.

Closed-form results are checked in checks/td2_7_learned_first_check.py; the review
log is ../reviews/td/td2_7_learned_first.md.
"""
from nbkit_td import answer, code, code_answer, markdown, reading, run

PHYS = "../../IliadPhysicsLearning/notebooks/second-draft"

CELLS = [
    # ================================================================ opening
    markdown(
        r"""
# NB2 §7 — What is learned first

**Training Dynamics, Iliad Intensive 2026**

Networks trained by gradient descent tend to use coarse structure in the data before
fine structure: class means and covariances before higher-order correlations, smooth
parts of a target before wiggly ones. This notebook works out three settings where that
order can be computed.

**Route.** The [Readings](#td27-readings) come first: every paper of the notebook,
grouped by section. The sections after them are the derivations and walkthrough
(exercises with worked solutions, and code), for whoever wants them.

- [7.1 Two moments](#td27-1): linear-in-input models see only means and covariances; small-weight networks see higher statistics later.
- [7.2 Higher cumulants](#td27-2): online SGD needs about $d^{\,k-1}$ samples to find a direction with information exponent $k\ge3$, far fewer if it is correlated with lower-order structure.
- [7.3 Spectral bias](#td27-3): a fixed kernel fits each eigenfunction at the rate of its eigenvalue; on the circle, low frequencies first.
"""
    ),
    # ================================================================ readings
    markdown(
        r"""
<a id="td27-readings"></a>

## Readings

**Two moments ([§7.1](#td27-1)).** The notebook's key paper.
"""
        + reading(
            "Neural networks trained with SGD learn distributions of increasing complexity",
            "https://arxiv.org/abs/2211.11567",
            "Refinetti, Ingrosso, Goldt (ICML 2023)",
            "The *distributional simplicity bias*: networks from two-layer nets to ResNets and ViTs, trained on Gaussian clones of CIFAR-10, match the real-data network's test accuracy early in training (clones from better generative models match it for longer), and a single-neuron expansion shows the stages: mean difference, Fisher discriminant, fourth-cumulant correction.",
            importance=5, length=3,
        )
        + r"""
**Higher cumulants ([§7.2](#td27-2)).** How many samples online SGD needs to find a direction hidden in a higher cumulant.
"""
        + reading(
            "Sliding down the stairs: how correlated latent variables accelerate learning with neural networks",
            "https://arxiv.org/abs/2404.08602",
            "Bardone, Goldt (ICML 2024)",
            "The mixed cumulant model, and nearly sharp online-SGD sample complexities for a single neuron: about $d\\log^2d$ for a covariance spike, $d^{\\,3}$ for a fourth-cumulant spike, and $d\\log^2d$ for both when their latent variables are correlated (Table 1; there $\\delta\\approx f(d)$ means $\\delta=o(f(d))$).",
            importance=4, length=3,
        )
        + reading(
            "Online stochastic gradient descent on non-convex losses from high-dimensional inference",
            "https://arxiv.org/abs/2003.10409",
            "Ben Arous, Gheissari, Jagannath (JMLR 22, 2021)",
            "Defines the information exponent and proves nearly sharp thresholds for online SGD to leave the uninformative equator (order $d$; between $d\\log d$ and $d\\log^2 d$; $d^{\\,k-1}$ up to logarithms); Section 2 computes the exponent for single-index models, GLMs, PCA and spiked tensors.",
            importance=3, length=3,
        )
        + r"""
**Spectral bias ([§7.3](#td27-3)).**
"""
        + reading(
            "The convergence rate of neural networks for learned functions of different frequencies",
            "https://arxiv.org/abs/1906.00425",
            "Basri, Jacobs, Kasten, Kritchman (NeurIPS 2019)",
            "The NTK spectrum of a two-layer ReLU network for uniform data on the sphere: without biases, odd frequencies $\\ge3$ cannot be learned; with biases, learning time $\\propto\\ell^2$ in 1D, which matches experiments with shallow and deep networks.",
            importance=4, length=3,
        )
        + reading(
            "On the spectral bias of neural networks",
            "https://arxiv.org/abs/1806.08734",
            "Rahaman et al. (ICML 2019)",
            "Six-layer ReLU networks fitting sums of sinusoids learn low frequencies first, even when higher frequencies have larger amplitude; low-frequency components are also more robust to parameter perturbations.",
            importance=3, length=2,
        )
        + r"""
**Also:**
[Székely, Bardone, Gerace, Goldt, *Learning from higher-order correlations, efficiently*](https://arxiv.org/abs/2312.14922)
(NeurIPS 2024; finite data reused for many epochs: about $d^{\,2}$ samples for low-degree
tests and for trained two-layer networks, where random features fail);
[Damian, Nichani, Ge, Lee, *Smoothing the landscape boosts the signal for SGD*](https://arxiv.org/abs/2305.10633)
(NeurIPS 2023; about $d^{\,k/2}$ samples with a smoothed loss);
[Xu et al., *Frequency Principle*](https://arxiv.org/abs/1901.06523) (2020; low-to-high
frequency fitting on MNIST and CIFAR-10).

Related, in this day: what basis a trained diffusion denoiser uses, and why it adapts to
image geometry (Kadkhodaie, Guth, Simoncelli, Mallat, ICLR 2024), is an exercise in
[`td2_8_further_readings` §3](td2_8_further_readings.html#td28-gahb).
"""
    ),
    # ================================================================ 7.1
    markdown(
        r"""
<a id="td27-1"></a>

## 7.1 Two moments are all a linear-in-input model sees

Which statistics of the data does gradient descent use first? A model linear in its
input $x$ sees only the class means and covariances. A network with small weights is
close to such a model, and higher statistics of $x$ enter its gradient at higher orders
in the weight scale.

**Paper:** Refinetti, Ingrosso & Goldt, *Neural networks trained with SGD learn
distributions of increasing complexity*, Section 2 (see [Readings](#td27-readings)).
They truncate a sigmoidal perceptron's gradient flow order by order in the weights: the
weight first aligns with the class-mean difference, then moves to Fisher's discriminant,
then picks up a fourth-cumulant correction.

*Core idea.* With square loss, a linear model's loss depends on the data only through
first and second moments. So a labelled dataset and its **Gaussian clone** (each class
replaced by the Gaussian with that class's mean and covariance) give identical
gradient-flow trajectories. Now take a two-layer network $f(x)=\sum_ja_j\sigma(w_j\cdot x)$
with all weights of order $\varepsilon$, and expand $\sigma$ in powers of the
pre-activation. The correlation of the label with a degree-$k$ function of the input,
$\mathbb E[y\,x^{\otimes k}]$, enters the gradient at order $\varepsilon^k$: the class-mean
difference first, then the covariance difference. This orders the *early* gradient; once
the weights grow, the terms become comparable.

<a id="td27-1-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Gradient flow $\dot\theta=-\nabla L$ in continuous time $t=k\eta$, with step
size $\eta$. Inputs $x\in\mathbb R^d$, labels $y$; for classification $y=\pm1$ with equal
probabilities, class means $\mu_\pm$, class covariances $C_\pm$. $\mathbb E$ averages over
the data, or over the training set when it is finite. $x^{\otimes k}$ is the $k$-fold
outer product, so $\mathbb E[x^{\otimes k}]$ is the $k$-th moment tensor.

*From NB1* (`td1_linear_networks`), restated. Square loss, whitened inputs
$\mathbb E[xx^\top]=I$, $\Sigma_{yx}=\mathbb E[yx^\top]=USV^\top$ with singular values
$s_\alpha$, and $a_\alpha$ the strength the model has learned in mode $\alpha$. A
*shallow* linear model $f=Wx$ has $\dot a_\alpha=s_\alpha-a_\alpha$: every mode relaxes at
the same rate, and $s_\alpha$ sets only the endpoint. A *two-layer* linear network
$f=W_2W_1x$ from small balanced initialization has $\dot a_\alpha=2a_\alpha(s_\alpha-a_\alpha)$,
a sigmoid that rises at $t\approx\frac1{2s_\alpha}\ln\frac{s_\alpha}{a_\alpha(0)}$, so
stronger modes are learned first. (NB1 writes $\tau\dot a$ with $\tau=1/\eta$ and counts
steps; with $t=k\eta$, $\tau$ drops out.)

**The linear model.** Take $f(x)=Wx+b$ with square loss. Append a constant to the input,
$\tilde x=(x,1)$, and write $\tilde W=[W,b]$. Then

$$
L(\tilde W)=\tfrac12\mathbb E\|y-\tilde W\tilde x\|^2
=\tfrac12\operatorname{tr}\mathbb E[yy^\top]-\operatorname{tr}\big(\tilde W\,\mathbb E[\tilde xy^\top]\big)
+\tfrac12\operatorname{tr}\big(\tilde W\,\mathbb E[\tilde x\tilde x^\top]\,\tilde W^\top\big).
$$

The data enter only through three moment matrices, so a dataset and its Gaussian clone
give *identical* trajectories, for any data distribution (with empirical moments on a
finite training set) and any optimizer whose update is a function of the full-batch
gradient. The fluctuations of minibatch SGD are different: their covariance involves
fourth moments of $x$ (Exercise 7.1.A).

<a id="td27-cumulants"></a>

**Cumulants** measure how far a distribution is from Gaussian at each order. For a random
vector $x$ with mean $\mu$, covariance $C$ and $\bar x=x-\mu$, the first cumulants are

$$
\kappa_1=\mu,\qquad \kappa_2=C,\qquad
\kappa_3=\mathbb E[\bar x^{\otimes3}],\qquad
\kappa_4^{ijkl}=\mathbb E[\bar x^i\bar x^j\bar x^k\bar x^l]-C^{ij}C^{kl}-C^{ik}C^{jl}-C^{il}C^{jk}.
$$

A Gaussian has $\kappa_p=0$ for all $p\ge3$, so the Gaussian clone of a class keeps its
$\kappa_1$ and $\kappa_2$ and sets all higher cumulants to zero. Cumulants add over
independent summands and scale as $\kappa_p(aX)=a^p\kappa_p(X)$.

**The test bed.** The demos use Bardone and Goldt's
[mixed cumulant model](#td27-mcm) in $d=16$: class $-1$ is $\mathcal N(0,I)$, and class $+1$
differs from it in its **mean** along $m$, its **variance** along $u$, and along $v$ **first
at the fourth cumulant** (same mean and variance, but bimodal). Bayes accuracies: 69.1% from
the means alone, 76.7% from means and covariances (the best possible on the Gaussian clone),
81.9% from everything (exact quadrature in
[`checks/td2_7_learned_first_check.py`](checks/td2_7_learned_first_check.py)).

**Small weights: the expansion.** Take $f(x)=\sum_{j=1}^Na_j\,\sigma(w_j\cdot x)$, with
$W$ the matrix of rows $w_j$ and an activation analytic at 0, $\sigma(z)=\sum_{k\ge1}c_kz^k$
($\tanh$, or GELU with $c_1=\tfrac12$, $c_2=1/\sqrt{2\pi}$, $c_3=0$). If all weights are of
order $\varepsilon$, the pre-activations are small and

$$
f(x)=c_1\,a^\top Wx\;+\;c_2\sum_ja_j(w_j\cdot x)^2\;+\;c_3\sum_ja_j(w_j\cdot x)^3+\dots
$$

The first term is NB1's two-layer linear network ($W_2=c_1a^\top$, $W_1=W$). Exercise 7.1.B
counts powers of $\varepsilon$ in the gradient. The label-weighted moment
$\mathbb E[y\,x^{\otimes k}]$, which for balanced $\pm1$ labels is half the class difference
of $k$-th moments, enters at order $\varepsilon^k$. The input moment $\mathbb E[x^{\otimes p}]$
enters at $\varepsilon^{p+1}$, because it reaches the gradient only through the output $f$,
which is itself small. So the class-mean difference leads (order $\varepsilon$), the
covariance difference follows (order $\varepsilon^2$, if $c_2\ne0$), and the linear
network's own feedback through $\Sigma_{xx}=\mathbb E[xx^\top]$ comes only at
$\varepsilon^3$. This orders the early gradient, not the whole course of learning, since
the weights grow and the terms become comparable. Refinetti et al. derive the staged order
for a single neuron; for larger networks it is observed rather than derived.

**Two limits of the expansion.** The NTK linearization
$f(x;\theta_0)+\nabla_\theta f(x;\theta_0)\cdot(\theta-\theta_0)$ is linear in the parameters
$\theta$ but not in $x$: it sees two moments of its *features* $\nabla_\theta f(x;\theta_0)$,
which involve all moments of $x$ when the pre-activations at initialization are of order
one (Exercise 7.1.C). And ReLU is positively homogeneous, so small weights do not make
$\mathrm{ReLU}(w\cdot x)$ linear in $x$ and the expansion does not apply; the staged order is
still observed in ReLU networks (demo 7.2.E, and Bardone and Goldt's experiments).

**Exercise 7.1.A — Linear models see two moments.**

1. With $\tilde x=(x,1)$ and $\tilde W=[W,b]$ as above, show that the loss $L=\tfrac12\mathbb E\|y-\tilde W\tilde x\|^2$ depends on the data only through $\mathbb E[yy^\top]$, $\mathbb E[\tilde xy^\top]$ and $\mathbb E[\tilde x\tilde x^\top]$, and derive the gradient flow for $\tilde W$.
2. Take centred data ($\mathbb E x=0$, $\mathbb E y=0$). Show that $b$ decouples, that $\dot W=\Sigma_{yx}-W\Sigma_{xx}$ with $\Sigma_{yx}=\mathbb E[yx^\top]$ and $\Sigma_{xx}=\mathbb E[xx^\top]$, and solve this in the eigenbasis of $\Sigma_{xx}$. Check that whitened inputs give NB1's $\dot a_\alpha=s_\alpha-a_\alpha$.
3. For balanced $\pm1$ labels, show that a dataset and its Gaussian clone have the same $\mathbb E[\tilde x\tilde x^\top]$, $\mathbb E[\tilde xy]$ and $\mathbb E[y^2]$. Conclude that gradient flow on a linear model gives the same trajectory on both.
4. Why does the conclusion of 3 fail for the fluctuations of minibatch SGD? Which moment of $x$ enters the covariance of a minibatch gradient?

<details><summary>Hint</summary>

For 2, write $W^*=\Sigma_{yx}\Sigma_{xx}^{-1}$ and $E=W-W^*$; the flow is $\dot E=-E\Sigma_{xx}$. For 3, $\mathbb E[xx^\top]=\sum_c\pi_c(C_c+\mu_c\mu_c^\top)$. For 4, the per-example gradient is $(\tilde W\tilde x-y)\tilde x^\top$.
</details>
"""
    ),
    answer(
        r"""
**1.** Expanding the square,

$$
\|y-\tilde W\tilde x\|^2=y^\top y-2\,y^\top\tilde W\tilde x+\tilde x^\top\tilde W^\top\tilde W\tilde x
=\operatorname{tr}(yy^\top)-2\operatorname{tr}(\tilde W\tilde xy^\top)+\operatorname{tr}(\tilde W\tilde x\tilde x^\top\tilde W^\top).
$$

Take expectations term by term; the expectation passes inside the trace and lands on $yy^\top$, $\tilde xy^\top$ and $\tilde x\tilde x^\top$. Here $\mathbb E[\tilde x\tilde x^\top]=\begin{psmallmatrix}\mathbb E[xx^\top]&\mathbb E x\\ \mathbb E x^\top&1\end{psmallmatrix}$, and $\mathbb E[\tilde xy^\top]$ stacks $\mathbb E[xy^\top]$ on $\mathbb E[y^\top]$. The gradient is

$$
\nabla_{\tilde W}L=-\big(\mathbb E[y\tilde x^\top]-\tilde W\,\mathbb E[\tilde x\tilde x^\top]\big),
\qquad
\dot{\tilde W}=\mathbb E[y\tilde x^\top]-\tilde W\,\mathbb E[\tilde x\tilde x^\top].
$$

The flow is a linear ODE whose coefficients are the moment matrices, so the trajectory from a given $\tilde W(0)$ is a function of those matrices alone. The same holds for full-batch GD and for any method whose update depends on the data only through $\nabla L$ (momentum, or Adam on full batches).

**2.** With $\mathbb E x=0$ the off-diagonal block of $\mathbb E[\tilde x\tilde x^\top]$ vanishes, so $\dot b=\mathbb E y-b=-b$ and $b\to0$. For $W$: $\dot W=\Sigma_{yx}-W\Sigma_{xx}$. Let $W^*=\Sigma_{yx}\Sigma_{xx}^{-1}$ (assume $\Sigma_{xx}\succ0$) and $E=W-W^*$. Then $\dot E=\Sigma_{yx}-(E+W^*)\Sigma_{xx}=-E\Sigma_{xx}$, so

$$
E(t)=E(0)\,e^{-\Sigma_{xx}t},\qquad E(t)\,e_i=e^{-\lambda_it}\,E(0)\,e_i .
$$

The error acting on input direction $e_i$ decays at rate $\lambda_i$, the input variance along $e_i$, so low-variance directions are learned slowly. This is the linear-model case of §7.3's spectral bias, with kernel $x\cdot x'$.

With $\Sigma_{xx}=I$ every rate is 1, $\dot W=\Sigma_{yx}-W$. With $u_\alpha,v_\alpha$ the $\alpha$-th columns of $U,V$ in $\Sigma_{yx}=USV^\top$ (NB1's notation; not the spikes of §7.2), $a_\alpha=u_\alpha^\top Wv_\alpha$ and $u_\alpha^\top\Sigma_{yx}v_\alpha=s_\alpha$, this projects to $\dot a_\alpha=s_\alpha-a_\alpha$, NB1's shallow result.

**3.** With class probabilities $\tfrac12$ and labels $y=\pm1$:

$$
\mathbb E[y^2]=1,\quad \mathbb E y=0,\quad
\mathbb E x=\tfrac12(\mu_++\mu_-),\quad
\mathbb E[yx]=\tfrac12(\mu_+-\mu_-),\quad
\mathbb E[xx^\top]=\tfrac12\textstyle\sum_\pm(C_\pm+\mu_\pm\mu_\pm^\top).
$$

Each is a function of the class means and covariances only, which the clone matches by construction. So the three moment matrices agree, and by part 1 so do the trajectories. On a finite training set the clone must match the *empirical* class means and covariances, as in the code below.

**4.** A minibatch gradient averages per-example gradients $g_i=(\tilde W\tilde x_i-y_i)\tilde x_i^\top$, and its covariance is $\frac1B\mathrm{Cov}(g_i)$. This contains terms like $\mathbb E[(\tilde W\tilde x)^2\,\tilde x\tilde x^\top]$, a fourth moment of $x$, in which a dataset and its clone generally differ. For a linear model with fresh batches the *expected* iterate still follows full-batch GD exactly, because the gradient is linear in $\tilde W$, but the fluctuations around it differ.
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(min(4, torch.get_num_threads()))
rng = np.random.default_rng(0)


def mcm_sample(N, d, spikes, beta_m=1.0, beta_u=5.0, beta_v=10.0, correlated=False,
               clone=False, gen=None):
    # Bardone & Goldt's mixed cumulant model (their eq. 1), labels y = +-1.
    #   y = -1:  x = z ~ N(0, I)
    #   y = +1:  x = beta_m m + sqrt(beta_u) lam u + Q (sqrt(beta_v) nu v + z)
    # lam ~ N(0,1); nu = +-1, independent of lam, or nu = sign(lam) if correlated.
    # Q = I - gamma v v^T whitens the v direction (their S).
    # clone=True replaces class +1 by the Gaussian with the same mean and covariance.
    m_, u, v = spikes
    gamma = 1 - 1 / np.sqrt(1 + beta_v)
    y = torch.randint(0, 2, (N,), generator=gen) * 2 - 1
    z = torch.randn(N, d, generator=gen)
    x_neg = torch.randn(N, d, generator=gen)
    lam = torch.randn(N, generator=gen)
    nu = torch.sign(lam) if correlated else (torch.randint(0, 2, (N,), generator=gen) * 2 - 1).to(lam.dtype)
    if clone:
        e_lam_nu = np.sqrt(2 / np.pi) if correlated else 0.0
        C = (torch.eye(d) + beta_u * torch.outer(u, u)
             + np.sqrt(beta_u * beta_v) * (1 - gamma) * e_lam_nu * (torch.outer(u, v) + torch.outer(v, u)))
        x_pos = beta_m * m_ + z @ torch.linalg.cholesky(C).T
    else:
        s = z + np.sqrt(beta_v) * nu[:, None] * v
        s = s - gamma * (s @ v)[:, None] * v
        x_pos = beta_m * m_ + np.sqrt(beta_u) * lam[:, None] * u + s
    return torch.where((y > 0)[:, None], x_pos, x_neg), y.to(lam.dtype)


def orthonormal_spikes(d, seed=2):
    q, _ = torch.linalg.qr(torch.randn(d, 3, generator=torch.Generator().manual_seed(seed)))
    return q.T  # rows: m, u, v
"""
    ),
    code(
        r"""
# Check 7.1.A.3: build an exact empirical clone of the test-bed data, and train a
# two-layer linear network on both.
torch.set_default_dtype(torch.float64)
d, n = 16, 2000
spikes = orthonormal_spikes(d)
gen = torch.Generator().manual_seed(0)
X, Y = mcm_sample(n, d, spikes, gen=gen)            # the test bed of 7.1.2
Xt, Yt = mcm_sample(5000, d, spikes, gen=gen)       # test set from the same distribution

def sym_sqrt(C, inv=False):
    e, V = torch.linalg.eigh(C)
    return V @ torch.diag(e.rsqrt() if inv else e.sqrt()) @ V.T

def empirical_clone(X, Y, gen):
    # Gaussian samples transformed to match each class's empirical mean and covariance exactly.
    Xc = torch.empty_like(X)
    for c in (-1.0, 1.0):
        idx = Y == c
        Xi = X[idx]
        Z = torch.randn(Xi.shape, generator=gen)
        Z = Z - Z.mean(0)
        Z = Z @ sym_sqrt(torch.cov(Z.T, correction=0), inv=True)
        Xc[idx] = Xi.mean(0) + Z @ sym_sqrt(torch.cov(Xi.T, correction=0))
    return Xc

Xc = empirical_clone(X, Y, gen)
print("max |E[xx^T] - E_clone[xx^T]|:", (X.T @ X - Xc.T @ Xc).abs().max().item() / n)
print("max |E[yx]   - E_clone[yx]|  :", (X.T @ Y - Xc.T @ Y).abs().max().item() / n)

check_times = sorted(set(np.unique(np.geomspace(1, 1500, 30).astype(int)).tolist() + [0]))

def train(Xtr, act, eps, width=32, lr=0.3, T=1500, seed=0):
    # Full-batch GD on 0.5*mean((a . act(W x) - y)^2) from weights of size eps.
    torch.manual_seed(seed)
    W = (eps * torch.randn(width, d) / np.sqrt(d)).requires_grad_()
    a = (eps * torch.randn(width) / np.sqrt(width)).requires_grad_()
    log = []
    for t in range(T + 1):
        if t in check_times:
            with torch.no_grad():
                pre = Xt @ W.T
                log.append(((act(pre) @ a * Yt > 0).double().mean().item(),
                            (act(pre) @ a).clone(), pre.pow(2).mean().sqrt().item()))
        if t == T:
            break
        loss = 0.5 * ((act(Xtr @ W.T) @ a - Y) ** 2).mean()
        gW, ga = torch.autograd.grad(loss, (W, a))
        with torch.no_grad():
            W -= lr * gW
            a -= lr * ga
    return log

identity = lambda z: z
lin_data, lin_clone = train(X, identity, 0.1), train(Xc, identity, 0.1)
print("two-layer linear network: max |f_data - f_clone| on the test set over training:",
      max((p[1] - q[1]).abs().max().item() for p, q in zip(lin_data, lin_clone)))
torch.set_default_dtype(torch.float32)
"""
    ),
    markdown(
        r"""
The empirical moments agree to rounding error, and so do the two linear networks'
outputs throughout training. The next exercise asks what changes when the network is
nonlinear.

**Exercise 7.1.B — Counting powers of the weight scale.**
Take $f(x)=\sum_ja_j\sigma(w_j\cdot x)$, $\sigma(z)=\sum_{k\ge1}c_kz^k$, square loss $L=\tfrac12\mathbb E(y-f)^2$, with all $a_j$ and $w_j$ of order $\varepsilon$.

1. Keep only the terms linear in the pre-activations, and show that this gives the gradients of the two-layer linear network $c_1a^\top Wx$. Its leading part is the drive $c_1\mathbb E[yx]$, at order $\varepsilon$. At what order does the linear network's feedback through $\Sigma_{xx}$ enter, compared with the $c_2$ term of the full network? In a scalar, balanced example, compare the initial growth rate with NB1's $\dot a=2a(s-a)$.
2. Write $\nabla_{w_j}L$ as a double series. Show that $\mathbb E[y\,x^{\otimes k}]$ enters with a prefactor of order $\varepsilon^k$, and the moment $\mathbb E[x^{\otimes p}]$ with a prefactor of order $\varepsilon^{p+1}$. Which activation coefficients must be nonzero for each term to appear?
3. Take an odd activation and no biases. Show that the gradient then contains only odd label-weighted moments $\mathbb E[y\,x^{\otimes k}]$ ($k$ odd) and even moments $\mathbb E[x^{\otimes p}]$. Conclude that two centred Gaussian classes that differ only in covariance cannot be told apart by such a network at all.
4. Suppose every class-conditional cumulant of order 3 vanishes, as in the test bed. Up to what order in $\varepsilon$ must the gradients on the data and on its Gaussian clone agree? At what order can they first differ, for GELU and for $\tanh$?

<details><summary>Hint</summary>

$\nabla_{w_j}L=-a_j\,\mathbb E\big[(y-f(x))\,\sigma'(w_j\cdot x)\,x\big]$ with $\sigma'(z)=\sum_kkc_kz^{k-1}$. Expand $f$ too. For 3, an odd $f$ satisfies $f(-x)=-f(x)$, and a centred Gaussian is symmetric under $x\mapsto-x$. For 4, in a class whose third cumulant vanishes, the moments up to order 3 are fixed by the mean and covariance.
</details>
"""
    ),
    answer(
        r"""
**1.** Keep the terms linear in the pre-activations: $\sigma(w_j\cdot x)\approx c_1w_j\cdot x$, $\sigma'\approx c_1$, $f\approx c_1a^\top Wx$. Then

$$
\nabla_{a_j}L\approx-c_1\big(\Sigma_{yx}-c_1a^\top W\Sigma_{xx}\big)w_j,
\qquad
\nabla_{w_j}L\approx-c_1a_j\big(\Sigma_{yx}-c_1a^\top W\Sigma_{xx}\big)^\top ,
$$

which are the gradients of $\tfrac12\mathbb E(y-c_1a^\top Wx)^2$. The drive $c_1\Sigma_{yx}$ is $O(\varepsilon)$; the feedback through $f$ is $O(\varepsilon^3)$, smaller than the $O(\varepsilon^2)$ nonlinear terms of part 2. If $\mathbb E[yx]=0$, which term leads depends on the first coefficient and statistic to survive. (With an odd activation and centrally symmetric classes, every odd label-weighted moment vanishes, and the $O(\varepsilon^3)$ feedback can remain the leading term.)

Scalar, whitened, balanced case: $f=c_1awx$, $s=\mathbb E[yx]$, $a=w$. With $q=c_1aw$, $\dot a=c_1w(s-q)$ and $\dot w=c_1a(s-q)$, so $\dot q=c_1^2(a^2+w^2)(s-q)=2c_1\,q\,(s-q)$. Only the early part is inside the expansion: the initial growth $q\propto e^{2c_1st}$, at rate $2c_1s$ instead of NB1's $2s$, so GELU ($c_1=\frac12$) starts this mode half as fast as a linear network with the same weights. The saturating part of NB1's sigmoid needs $q\approx s$, that is $aw=O(1)$, where the expansion has failed. For balanced $\pm1$ labels, $\Sigma_{yx}=\tfrac12(\mu_+-\mu_-)^\top$ has rank one, so the first direction learned is the mean difference.

**2.** Using the hint,

$$
\nabla_{w_j}L=-a_j\sum_{k\ge1}kc_k\,\mathbb E\big[y\,(w_j\cdot x)^{k-1}x\big]
\;+\;a_j\sum_{k\ge1}\sum_{l\ge1}kc_kc_l\sum_ia_i\,\mathbb E\big[(w_i\cdot x)^l(w_j\cdot x)^{k-1}x\big].
$$

The first sum contracts $\mathbb E[y\,x^{\otimes k}]$ with $k-1$ copies of $w_j$, and carries $a_jw_j^{k-1}$: order $\varepsilon^k$. It needs $c_k\ne0$. The second contracts the moment $\mathbb E[x^{\otimes p}]$ of order $p=k+l$, and carries $a_ja_iw_i^lw_j^{k-1}$: order $\varepsilon^{p+1}$. It needs some $c_kc_l\ne0$ with $k+l=p$. The series for $\nabla_aL$ gives the same orders.

| order | label-weighted statistic | input moment |
|---|---|---|
| $\varepsilon$ | $\mathbb E[yx]$ (needs $c_1$) | — |
| $\varepsilon^2$ | $\mathbb E[y\,x^{\otimes2}]$ (needs $c_2$) | — |
| $\varepsilon^3$ | $\mathbb E[y\,x^{\otimes3}]$ (needs $c_3$) | $\mathbb E[x^{\otimes2}]$ (needs $c_1^2$) |
| $\varepsilon^4$ | $\mathbb E[y\,x^{\otimes4}]$ (needs $c_4$) | $\mathbb E[x^{\otimes3}]$ (needs $c_1c_2$) |
| $\varepsilon^5$ | $\mathbb E[y\,x^{\otimes5}]$ (needs $c_5$) | $\mathbb E[x^{\otimes4}]$ (needs $c_1c_3$ or $c_2^2$) |

These are the earliest orders at which each statistic *can* appear. What has to be small is the pre-activation $w\cdot x$, not the weights as such; once it grows, all terms are comparable.

**3.** An odd $\sigma$ has $c_k=0$ for even $k$. In the first sum only odd $k$ survive, so only odd label-weighted moments appear. In the second sum $k$ and $l$ are both odd, so $p=k+l$ is even. The class difference of second moments, $\mathbb E[y\,x^{\otimes2}]=\tfrac12(C_+-C_-)$ for centred classes, never appears at any order.

For the classifier itself: an odd $f$ satisfies $f(-x)=-f(x)$. A centred Gaussian class is symmetric under $x\mapsto-x$, so $P_\pm(f(x)>0)=P_\pm(f(-x)>0)=P_\pm(f(x)<0)$. Each class is split evenly by the sign of $f$ (ignoring ties at $f=0$), and accuracy is exactly chance. This needs the symmetry of the class distributions; equal means alone are not enough (a zero-mean but skewed class can be partly separated by $\mathrm{sign}(x)$).

To see a covariance difference, the network needs an even part. With $c_2\ne0$, as for GELU (which is why the demo uses it), it enters at order $\varepsilon^2$; higher even coefficients see it later (for centred scalar Gaussian classes with standard deviations $s_\pm$, $\mathbb E[yx^4]=\tfrac32(s_+^4-s_-^4)$ enters through $c_4$ at $\varepsilon^4$). Biases also produce even terms: a bias of order $\varepsilon$ in $\tanh$ ($c_3\ne0$) gives a quadratic coefficient $\tfrac12\sigma''(b)\approx3c_3b=O(\varepsilon)$, so the covariance term appears at order $\varepsilon^3$.

**4.** The clone matches each class's mean and covariance. In a class with zero third cumulant, $\mathbb E_c[x^{\otimes3}]=\mu_c^{\otimes3}+3\,\mathrm{sym}(\mu_c\otimes C_c)$, the same as for the Gaussian with that mean and covariance. So all class-conditional moments up to order 3 agree, and with them every statistic in the table up to $\mathbb E[y\,x^{\otimes3}]$ and $\mathbb E[x^{\otimes3}]$. The gradients therefore agree through order $\varepsilon^3$, exactly as functions of the weights.

The fourth moments differ by the fourth cumulants. For GELU ($c_4\ne0$), the first possible difference is $\mathbb E[y\,x^{\otimes4}]$ at order $\varepsilon^4$. For $\tanh$ ($c_2=c_4=0$), it is order $\varepsilon^5$, through $\mathbb E[x^{\otimes4}]$ and $\mathbb E[y\,x^{\otimes5}]$. In the test bed, for example, with $x_m=m\cdot x$, $x_v=v\cdot x$ and $\kappa_4=\kappa_4(v\cdot x)$, $\mathbb E[y\,x_mx_v^4]=\tfrac12(3+\kappa_4)$ on the data and $\tfrac32$ on the clone, a difference of $\tfrac12\kappa_4\ne0$. Relative to the leading $O(\varepsilon)$ gradient, the difference starts at $O(\varepsilon^3)$ or $O(\varepsilon^4)$. The check script [`checks/td2_7_learned_first_check.py`](checks/td2_7_learned_first_check.py) confirms these orders by exact quadrature on the test bed: $\|\nabla L_{\rm data}-\nabla L_{\rm clone}\|$ scales as $\varepsilon^{3.97}$ for GELU and $\varepsilon^{4.98}$ for $\tanh$.
"""
    ),
    code(
        r"""
# A small-init GELU network trained by full-batch GD on the data and on its clone,
# both evaluated on real test data.
torch.set_default_dtype(torch.float64)
gelu = torch.nn.functional.gelu
g_data, g_clone = train(X, gelu, 0.1), train(Xc, gelu, 0.1)
gap = [(p[1] - q[1]).abs().mean().item() / max(p[1].abs().mean().item(), 1e-12)
       for p, q in zip(g_data, g_clone)]
tt = np.array(check_times[1:])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].semilogx(tt, [p[0] for p in g_data][1:], label="trained on data")
ax[0].semilogx(tt, [q[0] for q in g_clone][1:], "--", label="trained on clone")
for level, name in [(0.691, "Bayes, means only"), (0.767, "Bayes, means + covariances")]:
    ax[0].axhline(level, color="gray", lw=0.8, ls=":")
    ax[0].text(1.2, level + 0.004, name, fontsize=8)
ax[0].set(xlabel="GD step", ylabel="test accuracy (real data)", title="GELU net, small init")
ax[0].legend(fontsize=8, loc="lower right")
ax[1].loglog(tt, gap[1:], label="relative output gap, data vs clone")
ax[1].loglog(tt, [p[2] for p in g_data][1:], label="RMS pre-activation (trained on data)")
ax[1].set(xlabel="GD step", title="the gap opens as the pre-activations grow")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("relative gap at steps", [check_times[i] for i in (1, 10, 20, -1)], ":",
      [f"{gap[i]:.1e}" for i in (1, 10, 20, -1)])
torch.set_default_dtype(torch.float32)
"""
    ),
    markdown(
        r"""
Left: a two-stage staircase. Accuracy reaches about 0.68 within ~50 steps (the mean
difference), stalls, and rises around step 200 to about 0.73, above the 0.691 possible from
the means alone: the network now uses the variance difference along $u$, through GELU's
even part. The data- and clone-trained curves agree throughout, so within 1500 steps of
full-batch GD the $v$ signal gives no advantage (§7.2 computes how long online SGD needs).
Right: while the pre-activations are near their initial size, the two networks' outputs
differ by about $10^{-4}$ relative, consistent with 7.1.B.4 (on finite samples the
empirical clone also differs from the data in every moment beyond the second). Once the
pre-activations grow, the output gap reaches tens of per cent, while the accuracies stay
together.

**Exercise 7.1.C — What a linearized network sees: two moments of its features.**
This checks that a linearized (NTK) network is not limited to two moments of $x$.

1. For $f(x)=\sum_ja_j\sigma(w_j\cdot x)$ with parameters $\theta=(a_j,w_j)_j$ at initial values $\theta_0$, write down the tangent features $\phi(x)=\nabla_\theta f(x;\theta_0)$ and the empirical NTK $K(x,x')=\phi(x)\cdot\phi(x')$.
2. Show that the square loss of the linearized model $f_{\rm lin}(x;\theta)=f(x;\theta_0)+\phi(x)\cdot(\theta-\theta_0)$ depends on the data through $\mathbb E[\phi\phi^\top]$, $\mathbb E[\phi\,(y-f_0)]$ and $\mathbb E[(y-f_0)^2]$.
3. Take one input dimension and one unit, $\sigma=\tanh$, with feature $\phi_a(x)=\tanh(wx)$. Compare $x\sim\mathcal N(0,1)$ with $x$ uniform on $\{-1,+1\}$, which have the same mean and variance. Compute $\mathbb E[\tanh^2(wx)]$ for both at $w=1$, and expand in small $w$ to find the order at which they first differ.

<details><summary>Hint</summary>

$\tanh u=u-\tfrac{u^3}3+\tfrac{2u^5}{15}-\dots$, so $\tanh^2u=u^2-\tfrac23u^4+\tfrac{17}{45}u^6-\dots$ The Gaussian value at $w=1$ needs a numerical integral.
</details>
"""
    ),
    answer(
        r"""
**1.** $\partial f/\partial a_j=\sigma(w_j\cdot x)$ and $\partial f/\partial w_j=a_j\sigma'(w_j\cdot x)\,x$, so

$$
K(x,x')=\sum_j\sigma(w_j\cdot x)\,\sigma(w_j\cdot x')+\Big(\sum_ja_j^2\,\sigma'(w_j\cdot x)\,\sigma'(w_j\cdot x')\Big)\,x\cdot x' ,
$$

with all weights at their initial values.

**2.** $f_{\rm lin}$ is linear in $\theta$, with features $\phi(x)$ and offset $f_0(x)=f(x;\theta_0)$. With $\Delta=\theta-\theta_0$ and $r_0=y-f_0(x)$,

$$
\tfrac12\mathbb E(y-f_{\rm lin})^2=\tfrac12\mathbb E[r_0^2]-\Delta^\top\mathbb E[\phi\,r_0]+\tfrac12\Delta^\top\mathbb E[\phi\phi^\top]\Delta .
$$

This is 7.1.A with $x$ replaced by $\phi(x)$: the linearized model sees the first two moments of its *features*. For example, $\mathbb E[\phi_{a_i}\phi_{a_j}]=\mathbb E[\sigma(w_i\cdot x)\sigma(w_j\cdot x)]$ depends on the joint law of $(w_i\cdot x,w_j\cdot x)$, not just on the mean and covariance of $x$.

**3.** For $x=\pm1$, $\mathbb E\tanh^2(wx)=\tanh^2w$, which is $0.5800$ at $w=1$. For $x\sim\mathcal N(0,1)$, numerically $\mathbb E\tanh^2(x)=0.3943$. The same first two moments of $x$ give feature second moments 47% apart at an ordinary weight scale. For small $w$,

$$
\mathbb E\tanh^2(wx)=w^2\,\mathbb E x^2-\tfrac23w^4\,\mathbb E x^4+\tfrac{17}{45}w^6\,\mathbb E x^6-\dots
$$

Here $\mathbb E x^2=1$ for both, $\mathbb E x^4=3$ versus $1$, and $\mathbb E x^6=15$ versus $1$. The two agree at order $w^2$ and first differ at order $w^4$, by $-\tfrac43w^4$. (At $w=0.1$ the exact difference is $-1.282\times10^{-4}$, against $-1.280\times10^{-4}$ from the series.)

So what limits a model to two moments of $x$ is small pre-activations, not linearity in parameters. A lazy network at standard initialization has pre-activations of order one and uses higher moments from the start. With small initialization its features are nearly linear in $x$, and 7.1.B and this exercise give the same ordering.
"""
    ),
    # ================================================================ 7.2 (cumulants)
    markdown(
        r"""
<a id="td27-2"></a>

## 7.2 Higher cumulants: how long does a hidden direction take?

Suppose two classes have the same mean and variance along a direction $v$ in
$\mathbb R^d$, and differ there only in shape, in a higher cumulant. A model linear in its
input cannot see $v$ (§7.1). A neuron trained by online SGD can, but the number of
samples it needs grows as a power of $d$ set by the order at which $v$ first shows up in
the loss. Correlating $v$ with easier structure elsewhere cuts that cost sharply.

**Papers:** Bardone & Goldt, *Sliding down the stairs*; Ben Arous, Gheissari &
Jagannath, *Online SGD on non-convex losses from high-dimensional inference* (see
[Readings](#td27-readings)).

*Core idea.* Keep the neuron's weight $w$ on the unit sphere and track its overlap
$\alpha=w\cdot v$; a random start has $|\alpha|\approx d^{-1/2}$. If the population loss
near $\alpha=0$ first depends on $\alpha$ at order $\alpha^k$ (the **information
exponent** $k$), escaping that equator takes about $d$, $d\log d$ and $d^{\,k-1}$ samples
for $k=1$, $2$ and $\ge3$. A direction visible only in the fourth cumulant has $k=4$:
about $d^{\,3}$ samples. If its latent variable is correlated with that of a variance
direction $u$ (exponent 2), the loss gains a cross term $\alpha_u\alpha_v$ that drags $v$
along once $u$ is found, and $v$ then costs about $d\log^2d$ samples, like $u$.

<a id="td27-2-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* $d$ is the input dimension, $w$ a neuron's weight vector on the unit sphere,
$\alpha=w\cdot v$ its overlap with a unit direction $v$. The cumulants $\kappa_p$ are
defined in [§7.1's walkthrough](#td27-cumulants): $\kappa_1$ is the mean, $\kappa_2$ the
covariance, and for $p\ge3$ they vanish for a Gaussian, so a nonzero $\kappa_3$ or
$\kappa_4$ marks non-Gaussian shape.

**Hermite polynomials.** The loss of a single neuron on near-Gaussian inputs is best
expanded in Hermite polynomials; the degree of the first nonzero term sets how hard a
direction is to find. The (probabilists') Hermite polynomials $He_0=1$, $He_1=z$,
$He_2=z^2-1$, $He_3=z^3-3z$, $He_4=z^4-6z^2+3$, … are orthogonal under
$Z\sim\mathcal N(0,1)$, with $\mathbb E[He_j(Z)He_k(Z)]=k!\,\delta_{jk}$. A function, such
as an activation $\sigma$, expands as $\sigma(z)=\sum_k\frac{\sigma_k}{k!}He_k(z)$ with
$\sigma_k=\mathbb E[\sigma(Z)He_k(Z)]$. For ReLU, $\sigma_2=1/\sqrt{2\pi}$ and
$\sigma_4=-1/\sqrt{2\pi}$.

<a id="td27-mcm"></a>

**The mixed cumulant model** is the data model for §7.2 and for §7.1's test bed. Bardone
and Goldt put one signal each in the mean, the covariance and the fourth cumulant, with
strengths $\beta_m,\beta_u,\beta_v\ge0$. Class $y=-1$ is $x=z\sim\mathcal N(0,I_d)$. Class
$y=+1$ is

$$
x=\beta_m\,m+\sqrt{\beta_u}\,\lambda\,u+Q\big(\sqrt{\beta_v}\,\nu\,v+z\big),\qquad
\lambda\sim\mathcal N(0,1),\quad \nu=\pm1,\quad
Q=I-\frac{\beta_v}{1+\beta_v+\sqrt{1+\beta_v}}\,vv^\top ,
$$

with orthonormal spikes $m,u,v$ (they write $S$ for $Q$). The class means differ along
$m$ and the covariances along $u$. Along $v$, class $+1$ is a mixture of two Gaussians,
rescaled by $Q$ to unit variance: its mean and variance there match class $-1$, and its
fourth cumulant is negative (Exercise 7.2.A). With independent latents $\lambda,\nu$ the
Gaussian clone carries $m$ and $u$ but not $v$. §7.1's test bed has $\beta_m=1$,
$\beta_u=5$, $\beta_v=10$.

<a id="td27-infoexp"></a>

**The information exponent.**
Ben Arous, Gheissari and Jagannath study online SGD on the unit sphere, each sample used
once: $w\leftarrow\mathrm{normalize}\big(w-\tfrac\delta d\nabla_{\rm sph}\ell(w;x_t)\big)$,
where $\ell(w;x_t)$ is the loss on sample $x_t$, $\delta$ sets the step size, and
$\nabla_{\rm sph}=(I-ww^\top)\nabla$ is the gradient projected onto the sphere's tangent
space. The population loss depends on $w$ only through $\alpha$, $L(w)=\Phi(\alpha)$. The
information exponent is the smallest $k$ with $\Phi^{(k)}(0)\ne0$; that derivative must be
negative, so that the equator $\alpha=0$, where a random start sits, is unstable. For a
single-index target $y=h(v\cdot x)$ with Gaussian $x$, square loss and a student
$h(w\cdot x)$ of the same form, $k$ is the lowest degree $\ge1$ with a nonzero Hermite
coefficient of $h$. The number of samples needed to reach an overlap of order 1 scales as

| $k$ | samples |
|---|---|
| 1 | $d$ |
| 2 | $d\log d$ (proved: between $d\log d$ and $d\log^2d$) |
| $\ge3$ | $d^{\,k-1}$ (up to logarithms) |

For $k\ge2$ almost all of these samples go into this search phase; from an overlap of
order 1, any fixed accuracy costs $O(d/\delta)$ more. Exercise 7.2.B derives the powers of
$d$ from the dynamics of $\alpha$, which near the start is $\dot\alpha\propto\alpha^{k-1}$,
and 7.2.C checks them in simulation.

<details><summary>Precise statements, for reading the papers</summary>

*Weak recovery* means an overlap bounded away from 0 with probability $\to1$ as
$d\to\infty$; *strong recovery* means $\alpha\to1$. Ben Arous et al.'s Theorem 1.3 gives
strong recovery with $n\gg d$, $d\log^2d$ and $d^{\,k-1}\log^2d$ samples for $k=1$, $2$ and
$\ge3$, for suitably tuned steps and under assumptions on the loss, noise and
initialization; Theorem 1.4 shows weak recovery fails below the table's counts for small
enough steps. For $k\ge3$, online SGD on a *smoothed* loss needs only about $d^{\,k/2}$
samples ([Damian et al.](https://arxiv.org/abs/2305.10633)).
</details>

**Two hidden directions: sliding down the stairs.** Now the classes differ in variance
along $u$ and only in the fourth cumulant along $v$. Bardone and Goldt train a single
neuron $\sigma(w\cdot x)$ on the mixed cumulant model with no mean signal ($\beta_m=0$),
by online spherical SGD on the correlation loss $1-y\,\sigma(w\cdot x)$. The activation's
Hermite coefficients must satisfy $\sigma_2>0$, $\sigma_4<0$, plus some regularity
(smoothed ReLUs qualify). With overlaps $\alpha_u=w\cdot u$ and $\alpha_v=w\cdot v$,
Exercise 7.2.D derives the population loss near $\alpha_u=\alpha_v=0$,

$$
L\approx L_0-c_{20}\,\alpha_u^2-c_{11}\,\alpha_u\alpha_v-c_{04}\,\alpha_v^4 ,
$$

with a constant $L_0$, coefficients $c_{20},c_{04}>0$, and $c_{11}\propto\mathbb E[\lambda\nu]$,
the correlation of the latent variables $\lambda$ (along $u$) and $\nu=\pm1$ (along $v$).
So $u$ has information exponent 2 and $v$ has exponent 4. Their results:

| setting | samples to find $v$ |
|---|---|
| cumulant spike alone | about $d^{\,3}$ (up to logarithms) |
| both spikes, independent latents | not fewer than about $d^{\,3}$ (only this lower bound is proved; learning $u$ can even push $\alpha_v$ back toward 0) |
| both spikes, correlated latents, $\mathbb E[\lambda\nu]>0$ | about $d\log^2d$, the same as for $u$ |

Correlation adds a term $c_{11}\alpha_u$ to $\dot\alpha_v$ that does not vanish as
$\alpha_v\to0$, so $v$ is dragged along once $u$ is found. It also makes $v$ visible at
second order: the class covariance gains a $u$–$v$ cross term (Exercise 7.2.A.3). Their
two-layer ReLU networks show the same staircase (mean, then covariance, then cumulant
direction), and correlating the latents removes the last plateau; demo 7.2.E reproduces this
at small $d$.

<details><summary>Precise statements, for reading the paper</summary>

Their Propositions 1–4 and Table 1 state these as weak-recovery results, with ranges of
the step parameter $\delta$ (step $\delta/d$; Table 1's $\delta\approx f(d)$ means
$\delta=o(f(d))$). The lower bounds hold for small steps ($\delta=o(1/d)$ for the cumulant
spike), the sufficient conditions need suitably tuned steps, and the correlated-latent
result assumes $\alpha_u$ and $\alpha_v$ start with the same sign.
</details>

*Scope.* These results are for online SGD (fresh samples at every step) on a single
neuron. With a finite dataset reused for many epochs the counting changes; see Székely et
al. in the [Readings](#td27-readings).

**Exercise 7.2.A — The mixed cumulant model, by hand.**
Class $+1$ is $x=\beta_m m+\sqrt{\beta_u}\lambda u+Q(\sqrt{\beta_v}\nu v+z)$ with
orthonormal $m,u,v$, $\lambda\sim\mathcal N(0,1)$, $\nu=\pm1$, $z\sim\mathcal N(0,I_d)$ and
$Q=I-\gamma vv^\top$, $\gamma=\frac{\beta_v}{1+\beta_v+\sqrt{1+\beta_v}}$. Write
$s=\sqrt{\beta_v}\,\nu+\zeta$, with $\zeta=v\cdot z\sim\mathcal N(0,1)$, for the
$v$-component before whitening.

1. Show that $Q$ maps $v\mapsto v/\sqrt{1+\beta_v}$ and fixes $v^\perp$. Conclude that $v\cdot x=s/\sqrt{1+\beta_v}$ has mean 0 and variance 1 in class $+1$, as in class $-1$.
2. Compute the third and fourth cumulants of $v\cdot x$ in class $+1$. Evaluate the fourth at $\beta_v=10$.
3. Now let $\nu=\mathrm{sign}(\lambda)$. Compute the class-$+1$ covariance and show it has a cross term $c\,(uv^\top+vu^\top)$. Find $c$, and evaluate it at $\beta_u=5$, $\beta_v=10$.

<details><summary>Hint</summary>

For a Rademacher variable $\mathbb E\nu^4=1$, so $\kappa_4(\nu)=1-3=-2$. For 3, $\mathbb E[\lambda\,\mathrm{sign}\lambda]=\mathbb E|\lambda|$.
</details>
"""
    ),
    answer(
        r"""
**1.** Write $\gamma=\frac{\beta_v}{1+\beta_v+\sqrt{1+\beta_v}}$. Then $Qv=(1-\gamma)v$ and $Qw=w$ for $w\perp v$. With $r=\sqrt{1+\beta_v}$,

$$
1-\gamma=\frac{r^2+r-(r^2-1)}{r^2+r}=\frac{r+1}{r(r+1)}=\frac1r .
$$

So $v\cdot x=v\cdot Q(\sqrt{\beta_v}\nu v+z)=\frac{\sqrt{\beta_v}\nu+\zeta}{\sqrt{1+\beta_v}}$, using $m,u\perp v$. Its mean is 0 and its variance is $\frac{\beta_v\mathbb E\nu^2+1}{1+\beta_v}=1$. The components of $x$ orthogonal to $v$ are untouched by $Q$, so the class-$+1$ covariance is $I+\beta_uuu^\top$, with nothing along $v$.

**2.** $v\cdot x=\frac{\sqrt{\beta_v}}{r}\nu+\frac1r\zeta$ is a sum of independent terms. Cumulants of order $\ge3$ vanish for the Gaussian $\zeta$, and odd cumulants vanish for the symmetric $\nu$. So $\kappa_3=0$ and

$$
\kappa_4(v\cdot x)=\Big(\frac{\sqrt{\beta_v}}{r}\Big)^4\kappa_4(\nu)=-\frac{2\beta_v^2}{(1+\beta_v)^2}.
$$

At $\beta_v=10$ this is $-200/121\approx-1.65$. Along $v$, class $+1$ is a mixture of two Gaussian bumps at $\pm\sqrt{\beta_v/(1+\beta_v)}\approx\pm0.95$, each of standard deviation $1/\sqrt{11}\approx0.30$: unit variance, strongly platykurtic. The Bayes classifier using only $v\cdot x$ reaches about 71% accuracy, but no statistic of order 1 or 2 shows anything.

**3.** The class-$+1$ input minus its mean is $\sqrt{\beta_u}\lambda u+Q(\sqrt{\beta_v}\nu v+z)$. Since $z$ is independent of $\lambda,\nu$, its covariance is

$$
\beta_uuu^\top+Q\big(\beta_vvv^\top+I\big)Q+\sqrt{\beta_u\beta_v}\,\mathbb E[\lambda\nu]\,\big(u(Qv)^\top+(Qv)u^\top\big).
$$

By part 1, $Q(\beta_vvv^\top+I)Q=I$ and $Qv=v/\sqrt{1+\beta_v}$. With $\nu=\mathrm{sign}\lambda$, $\mathbb E[\lambda\nu]=\mathbb E|\lambda|=\sqrt{2/\pi}$, so

$$
\mathrm{Cov}_+=I+\beta_uuu^\top+c\,(uv^\top+vu^\top),\qquad c=\sqrt{\frac{2}{\pi}}\sqrt{\frac{\beta_u\beta_v}{1+\beta_v}} .
$$

At $\beta_u=5$, $\beta_v=10$: $c=0.798\times2.132=1.70$. The top eigenvector of $\mathrm{Cov}_+-I$ now tilts toward $v$, so the Gaussian clone of the correlated model contains information about $v$, and finding it is no longer purely a fourth-order problem. (`mcm_sample(..., clone=True, correlated=True)` builds the clone with this cross term.)
"""
    ),
    markdown(
        r"""
**Exercise 7.2.B — The powers of $d$ from the overlap dynamics.**
Where do the powers of $d$ in the [information-exponent table](#td27-infoexp) come from? Take online spherical SGD with step $\delta/d$, a population loss with $\Phi(\alpha)\approx-c\,\alpha^k$ near 0, and per-sample gradients of norm about $\sqrt d$ (as for $x\sim\mathcal N(0,I_d)$) with an $O(1)$ component along $v$. Track $\alpha_t=w_t\cdot v$ from $\alpha_0\approx d^{-1/2}$.

1. Write the one-step change of $\alpha$ as drift + noise + a correction from renormalizing $w$. Estimate each: the drift from $-\Phi'$, the noise from the $v$-component of the sample gradient, and the renormalization from $\|w-\tfrac\delta dg\|^2=1+\delta^2\|g\|^2/d^2$.
2. Require that the drift beat the renormalization at $\alpha\approx\alpha_0$. What does that say about $\delta$ for $k=2$ and for $k\ge3$?
3. Integrate the drift from $\alpha_0$ to order 1 to get the escape time in steps, for $k=1$, $k=2$, and $k\ge3$ with the largest allowed $\delta$.
4. Compare the noise accumulated during the escape with the starting overlap. Is the noise a problem?

<details><summary>Hint</summary>

The noise in $\alpha$ is a martingale with increments of size $\delta/d$. For $k\ge3$, $\int_{\alpha_0}d\alpha/\alpha^{k-1}\approx\alpha_0^{-(k-2)}/(k-2)$. For $k=2$, the linearized equation $d\alpha=a\alpha\,dt+b\,dB$ has solution $e^{at}\big(\alpha_0+\int_0^te^{-as}b\,dB_s\big)$.
</details>
"""
    ),
    answer(
        r"""
**1.** Let $g$ be the spherical sample gradient, $\tilde w=w-\frac\delta dg$ and $w'=\tilde w/\|\tilde w\|$. Then

$$
\alpha'=\frac{\alpha-\frac\delta d\,g\cdot v}{\sqrt{1+\delta^2\|g\|^2/d^2}}
\approx\alpha\;\underbrace{-\;\frac\delta d\,\mathbb E[g\cdot v]}_{\text{drift}}\;\underbrace{-\;\frac\delta d\,(g\cdot v-\mathbb E[g\cdot v])}_{\text{noise}}\;\underbrace{-\;\frac{\delta^2\|g\|^2}{2d^2}\,\alpha}_{\text{renormalization}} .
$$

The drift is $-\frac\delta d\Phi'(\alpha)(1-\alpha^2)\approx\frac\delta d\,ck\,\alpha^{k-1}$. The noise has mean zero and size $\frac\delta d\cdot O(1)$. With $\|g\|^2\approx d$, the renormalization is $\approx-\frac{\delta^2}{2d}\alpha$, which always pulls $\alpha$ toward 0.

**2.** Drift $\gg$ renormalization at $\alpha_0$ means $\frac\delta d\alpha_0^{k-1}\gg\frac{\delta^2}{d}\alpha_0$, that is, $\delta\ll\alpha_0^{k-2}=d^{-(k-2)/2}$. For $k=1$ the drift $\frac\delta dck$ does not depend on $\alpha$, and any $\delta=O(1)$ works. For $k=2$ the condition is $\delta\ll1$. For $k\ge3$ it is $\delta\ll d^{-(k-2)/2}$, a step that shrinks with dimension.

**3.** The drift ODE is $\frac{d\alpha}{dt}=\frac{\delta ck}{d}\alpha^{k-1}$, with $t$ counting steps, which equals samples.

- $k=1$: $\alpha$ grows linearly and reaches order 1 after $t\sim d/\delta\sim d$.
- $k=2$: $\alpha=\alpha_0e^{2\delta ct/d}$, which reaches order 1 at $t\approx\frac{d}{2c\delta}\ln\frac1{\alpha_0}=\frac{d\ln d}{4c\delta}$. With $\delta$ a small constant, this is $\sim d\log d$, the lower bound in the table.
- $k\ge3$: $t\approx\frac{d}{\delta ck(k-2)}\alpha_0^{-(k-2)}=\frac{d^{\,k/2}}{\delta ck(k-2)}$. With $\delta\sim d^{-(k-2)/2}$, at the edge of part 2, $t\sim d^{\,k/2}\cdot d^{(k-2)/2}=d^{\,k-1}$.

**4.** For $k=2$, linearize: $d\alpha=a\alpha\,dt+b\,dB$ with $a\asymp\delta/d$ and $b\asymp\delta/d$. The solution is $\alpha_t=e^{at}\big(\alpha_0+\int_0^te^{-as}b\,dB_s\big)$, so the noise acts as a random shift of the starting point, with variance $\frac{b^2}{2a}(1-e^{-2at})=O(\delta/d)$. That is small compared with $\alpha_0^2=1/d$ when $\delta\ll1$, and even if it were comparable it would only change $\ln(1/\alpha_0)$ by $O(1)$. So noise costs nothing extra at the level of powers of $d$. (Choosing $\delta\sim1/\ln d$, as Bardone and Goldt do, is one way to make a rigorous proof work, at a cost of $d\log^2d$.)

For $k\ge3$, do the same comparison at the start: the local growth rate is $a=\frac\delta dck\alpha_0^{k-2}$, and the effective noise variance is $b^2/2a\sim\frac{\delta}{d\,\alpha_0^{k-2}}$. Requiring it to be below $\alpha_0^2$ gives $\delta\ll d\,\alpha_0^k=d^{-(k-2)/2}$, the same condition as part 2. So noise and renormalization together set the largest usable step, and at that step the drift takes $d^{\,k-1}$ steps to act. This is a heuristic for the powers of $d$; the theorems control logarithmic factors that this argument does not determine.

For the fourth-cumulant spike, $k=4$ gives $d^{\,3}$, Bardone and Goldt's Proposition 2.
"""
    ),
    markdown(
        r"""
**Exercise 7.2.C — Code: escape times versus dimension.**
Simulate the setting of 7.2.B. Use inputs $x\sim\mathcal N(0,I_d)$, labels $y=h_k(v\cdot x)$ with $h_k=He_k/\sqrt{k!}$, a student $h_k(w\cdot x)$, and the correlation loss $1-y\,h_k(w\cdot x)$. Its population loss is $1-\alpha^k$, so the information exponent is $k$. Run online spherical SGD (one fresh sample per step, step $\delta/d$) from a random start with $\alpha_0>0$, for $k=1,2,3$ and $d\in\{16,32,64\}$. Use $\delta=1$, $1/\ln d$ and $d^{-1/2}/\ln d$ respectively. Record the first time $\alpha>1/2$ (for odd $k$ the sign matters: at $\alpha=-\tfrac12$ the loss $1-\alpha^3$ is worse than at the start; for even $k$ use $|\alpha|>1/2$). Take the median over a few dozen runs, and compare with $d$, $d\ln^2d$ and $d^{\,2}\ln d$. Vectorize over runs.

<details><summary>Hint</summary>

For one sample, the gradient of $1-y\,h_k(w\cdot x)$ is $-y\,h_k'(w\cdot x)\,x$. Remove its radial part, $g\mapsto g-(g\cdot w)w$, step by $-\frac\delta dg$, and renormalize $w$. With $\alpha_0\approx d^{-1/2}$, which is $0.25$ at $d=16$, the threshold $1/2$ is not far away, so at these sizes the search phase is short, especially for $k=1$.
</details>
"""
    ),
    code_answer(
        r"""
H = {1: (lambda z: z, lambda z: np.ones_like(z)),
     2: (lambda z: (z * z - 1) / np.sqrt(2), lambda z: np.sqrt(2) * z),
     3: (lambda z: (z**3 - 3 * z) / np.sqrt(6), lambda z: (3 * z * z - 3) / np.sqrt(6))}

def escape_times(k, d, delta, runs=48, t_max=10**6):
    # Your code: online spherical SGD on 1 - y h_k(w.x), y = h_k(x_0) (teacher v = e_0).
    # Return an array of first times w_0 > 1/2 (odd k) or |w_0| > 1/2 (even k); np.inf if not reached.
    ...
""",
        r"""
H = {1: (lambda z: z, lambda z: np.ones_like(z)),
     2: (lambda z: (z * z - 1) / np.sqrt(2), lambda z: np.sqrt(2) * z),
     3: (lambda z: (z**3 - 3 * z) / np.sqrt(6), lambda z: (3 * z * z - 3) / np.sqrt(6))}

def escape_times(k, d, delta, runs=48, t_max=10**6):
    h, dh = H[k]
    w = rng.standard_normal((runs, d))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    w[:, 0] = np.abs(w[:, 0])                     # start in the upper half-sphere
    T = np.full(runs, np.inf)
    for t in range(1, t_max + 1):
        x = rng.standard_normal((runs, d))
        g = (h(x[:, 0]) * dh((w * x).sum(1)))[:, None] * x   # = -grad of the correlation loss
        g -= (g * w).sum(1)[:, None] * w                       # spherical projection
        w += (delta / d) * g
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        a0 = w[:, 0] if k % 2 else np.abs(w[:, 0])           # odd k: the sign matters
        T[np.isinf(T) & (a0 > 0.5)] = t
        if not np.isinf(T).any():
            break
    return T

dims = [16, 32, 64]
law = {1: lambda d: d, 2: lambda d: d * np.log(d) ** 2, 3: lambda d: d**2 * np.log(d)}
step = {1: lambda d: 1.0, 2: lambda d: 1 / np.log(d), 3: lambda d: d**-0.5 / np.log(d)}
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for k in (1, 2, 3):
    med = [float(np.median(escape_times(k, d, step[k](d)))) for d in dims]
    slope = np.polyfit(np.log(dims), np.log(med), 1)[0]
    slope_law = np.polyfit(np.log(dims), np.log([law[k](d) for d in dims]), 1)[0]
    print(f"k={k}: median escape times {med}, ratio to the law "
          f"{np.round(np.array(med) / [law[k](d) for d in dims], 2)}, "
          f"log-log slope {slope:.2f} (the law's slope over this range: {slope_law:.2f})")
    ax.loglog(dims, med, "o-", label=f"k = {k} (slope {slope:.2f})")
ax.set(xlabel="dimension d", ylabel=r"median samples to reach $\alpha>1/2$",
       title="online SGD: search time vs information exponent")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""",
    ),
    markdown(
        r"""
The fitted slopes land within a few tenths of the laws' slopes over $d=16$ to $64$ (1.00,
1.58, 2.29), and the ratios to the laws stay roughly constant: a finite-size consistency
check of the powers at the theory's step sizes, not a measurement of exponents or log
factors. That larger steps fail needs a separate sweep. In one (at $d=64$, $k=3$, 32 runs
per point, $\delta$ from 0.03 to 10), the median escape time was lowest for
$\delta\approx0.3$–$1$ and about 25 times longer at $\delta=10$, where 44% of runs had not
escaped after 60000 steps. A $k=4$ run would need about $d^{\,3}$ steps, which is why demo
7.2.E uses small $d$.

**Exercise 7.2.D — A single neuron's population loss, in Hermite form.**
This derives the two-direction loss above. Take the mixed cumulant model with
$\beta_m=0$, balanced classes, a unit vector $w$ with overlaps $\alpha_u=w\cdot u$,
$\alpha_v=w\cdot v$, and the correlation loss $L(w)=1-\mathbb E[y\,\sigma(w\cdot x)]$.
Write $X=w\cdot x$, $\mathbb E_+$ for the expectation over class $+1$, and
$\sigma_k=\mathbb E[\sigma(Z)He_k(Z)]$ for the Hermite coefficients.

1. Show that $L=1-\tfrac12\big(\mathbb E_+[\sigma(X)]-\mathbb E[\sigma(Z)]\big)$ with $Z\sim\mathcal N(0,1)$, and that, expanding $\sigma$ in Hermite polynomials, $L=1-\tfrac12\sum_{k\ge1}\frac{\sigma_k}{k!}\,\mathbb E_+[He_k(X)]$.
2. Under class $+1$, show that $X$ has mean 0 and $\mathrm{Var}(X)=1+\beta_u\alpha_u^2+2c\,\alpha_u\alpha_v$, with $c$ from 7.2.A.3 ($c=0$ for independent latents), and that its odd cumulants vanish.
3. Use $\mathbb E[He_2(X)]=\mathrm{Var}(X)-1$ and $\mathbb E[He_4(X)]=\kappa_4(X)+3(\mathrm{Var}(X)-1)^2$ for a mean-zero $X$. For independent latents, $\kappa_4(X)=\alpha_v^4\,\kappa_4(v\cdot x)$. Derive
$$
L\approx L_0-\underbrace{\tfrac{\sigma_2\beta_u}{4}}_{c_{20}}\alpha_u^2-\underbrace{\tfrac{\sigma_2c}{2}}_{c_{11}}\alpha_u\alpha_v-\underbrace{\tfrac{\sigma_4\kappa_4(v\cdot x)}{48}}_{c_{04}}\alpha_v^4
$$
to leading order in each term, and read off the information exponents of $u$ and $v$. Why do the signs $\sigma_2>0$, $\sigma_4<0$ matter?
4. Ignoring noise and the spherical constraint, gradient flow on this $L$ is $\dot\alpha_u=2c_{20}\alpha_u+c_{11}\alpha_v$, $\dot\alpha_v=c_{11}\alpha_u+4c_{04}\alpha_v^3$, from $|\alpha_u(0)|,|\alpha_v(0)|\sim d^{-1/2}$. Show that with $c_{11}>0$, $\alpha_v$ grows on the same time scale as $\alpha_u$, and find the ratio $\alpha_v/\alpha_u$ it approaches, while with $c_{11}=0$ it reaches order 1 only after a time of order $d/c_{04}$.

<details><summary>Hint</summary>

For 1, class $-1$ has $X\sim\mathcal N(0,1)$ because $\|w\|=1$, and $\mathbb E[He_k(Z)]=0$ for $k\ge1$. For 2, $X=\sqrt{\beta_u}\alpha_u\lambda+\frac{\alpha_v}{r}\sqrt{\beta_v}\nu+G$, where $G$ is Gaussian with variance $1-\alpha_v^2+\alpha_v^2/r^2$ and $r=\sqrt{1+\beta_v}$. For 4, the linear part of the flow is $\dot\alpha=M\alpha$ with $M=\begin{psmallmatrix}2c_{20}&c_{11}\\c_{11}&0\end{psmallmatrix}$.
</details>
"""
    ),
    answer(
        r"""
**1.** $L=1-\tfrac12\mathbb E_+[\sigma(X)]+\tfrac12\mathbb E_-[\sigma(X)]$. Under class $-1$, $x\sim\mathcal N(0,I)$ and $\|w\|=1$, so $X\sim\mathcal N(0,1)$ and $\mathbb E_-[\sigma(X)]=\mathbb E[\sigma(Z)]$. Insert $\sigma=\sum_k\frac{\sigma_k}{k!}He_k$ into $\mathbb E_+[\sigma(X)]-\mathbb E[\sigma(Z)]$. The $k=0$ terms cancel, and $\mathbb E[He_k(Z)]=0$ for $k\ge1$, which leaves $\sum_{k\ge1}\frac{\sigma_k}{k!}\mathbb E_+[He_k(X)]$. (Exchanging sum and expectation is valid as an expansion near the equator, where the law of $X$ is close to $\mathcal N(0,1)$; Bardone and Goldt do it rigorously with the likelihood ratio, their eq. 6–7.)

**2.** With $x=\sqrt{\beta_u}\lambda u+Q(\sqrt{\beta_v}\nu v+z)$ and $Q$ symmetric, $X=\sqrt{\beta_u}\alpha_u\lambda+(Qw)\cdot(\sqrt{\beta_v}\nu v+z)$. Here $(Qw)\cdot v=\alpha_v/r$, and $(Qw)\cdot z$ is Gaussian with variance $\|Qw\|^2=1-\alpha_v^2+\alpha_v^2/r^2$. So

$$
X=\sqrt{\beta_u}\,\alpha_u\lambda+\frac{\sqrt{\beta_v}\,\alpha_v}{r}\,\nu+G,
$$

with mean 0. Its variance is $\beta_u\alpha_u^2+\frac{\beta_v\alpha_v^2}{r^2}+1-\alpha_v^2+\frac{\alpha_v^2}{r^2}+2\sqrt{\beta_u\beta_v}\,\frac{\alpha_u\alpha_v}{r}\mathbb E[\lambda\nu]=1+\beta_u\alpha_u^2+2c\,\alpha_u\alpha_v$, because $\frac{\beta_v+1}{r^2}=1$. The law of $(\lambda,\nu,G)$ is invariant under $(\lambda,\nu,G)\mapsto-(\lambda,\nu,G)$, also when $\nu=\mathrm{sign}\lambda$, so $X$ is symmetric and its odd cumulants vanish.

**3.** Odd $k$ contribute nothing, since $He_k$ is odd and $X$ is symmetric. For $k=2$: $\frac{\sigma_2}{2}(\beta_u\alpha_u^2+2c\alpha_u\alpha_v)$. For $k=4$, with independent latents, $\lambda$ and $G$ are Gaussian and $\nu$ is independent, so $\kappa_4(X)=\big(\frac{\sqrt{\beta_v}\alpha_v}{r}\big)^4\kappa_4(\nu)=\alpha_v^4\kappa_4(v\cdot x)$, by 7.2.A.2. Then $\frac{\sigma_4}{24}\big(\alpha_v^4\kappa_4(v\cdot x)+3(\beta_u\alpha_u^2)^2\big)$. The second piece is $O(\alpha_u^4)$, a correction to the $\alpha_u^2$ term. Multiplying by $-\tfrac12$ gives

$$
L\approx L_0-\frac{\sigma_2\beta_u}{4}\alpha_u^2-\frac{\sigma_2c}{2}\alpha_u\alpha_v-\frac{\sigma_4\kappa_4(v\cdot x)}{48}\alpha_v^4 .
$$

This is Bardone and Goldt's eq. 9, with explicit coefficients. With $\alpha_v=0$ the first nonzero term in $\alpha_u$ is quadratic, so $u$ has exponent 2. With $\alpha_u=0$ the first in $\alpha_v$ is quartic, so $v$ has exponent 4. Escaping the equator needs these terms to *decrease* $L$, as in Ben Arous et al.'s Definition 1.2. $\sigma_2>0$ makes the $\alpha_u^2$ term negative. $\kappa_4(v\cdot x)<0$ together with $\sigma_4<0$ makes the $\alpha_v^4$ term negative; with the opposite sign of $\sigma_4$, the equator would be locally stable along $v$. So the exponent is a property of the data *and* the activation and loss. A linear student, with $\sigma_k=0$ for $k\ge2$, never sees $v$. For ReLU, $\sigma_2=-\sigma_4=1/\sqrt{2\pi}$, and both signs are right. The check script [`checks/td2_7_learned_first_check.py`](checks/td2_7_learned_first_check.py) confirms the three coefficients against an exact computation of $L$ for ReLU.

**4.** The linear part is $\dot\alpha=M\alpha$ with $M=\begin{psmallmatrix}2c_{20}&c_{11}\\c_{11}&0\end{psmallmatrix}$. Its eigenvalues solve $\lambda^2-2c_{20}\lambda-c_{11}^2=0$, so $\lambda_\pm=c_{20}\pm\sqrt{c_{20}^2+c_{11}^2}$, with $\lambda_+\ge2c_{20}>0\ge\lambda_-$. The growing eigenvector is $(\lambda_+,c_{11})$; check: $M(\lambda_+,c_{11})^\top=(2c_{20}\lambda_++c_{11}^2,\;c_{11}\lambda_+)^\top=\lambda_+(\lambda_+,c_{11})^\top$.

- If $c_{11}>0$, a generic start has a component of size $\sim d^{-1/2}$ along this mode, so $(\alpha_u,\alpha_v)\approx d^{-1/2}e^{\lambda_+t}(\lambda_+,c_{11})/\|\cdot\|$. Both overlaps grow at the same rate, with $\alpha_v/\alpha_u\to c_{11}/\lambda_+=c_{11}/(c_{20}+\sqrt{c_{20}^2+c_{11}^2})$, which is $\approx c_{11}/(2c_{20})$ for weak correlation. $\alpha_u$ reaches order 1 at $t\approx\frac{\ln d}{2\lambda_+}$, with $\alpha_v$ then of order $c_{11}/\lambda_+$; from there the cubic term (same sign) takes over. A start with $\alpha_u,\alpha_v$ of the same sign projects onto $(\lambda_+,c_{11})$ with that common sign, so the growing mode carries both overlaps away from 0 together; this is the sign condition in Bardone and Goldt's Proposition 4. (The loss is invariant under $\alpha\mapsto-\alpha$, so a negative common sign gives the mirror image.)
- If $c_{11}=0$ the equations decouple. $\alpha_u$ grows as $e^{2c_{20}t}$ and reaches order 1 at $\frac{\ln d}{4c_{20}}$. $\dot\alpha_v=4c_{04}\alpha_v^3$ gives $\alpha_v^{-2}(t)=\alpha_v(0)^{-2}-8c_{04}t$, which blows up at $t=\frac{1}{8c_{04}\alpha_v(0)^2}\sim\frac{d}{8c_{04}}$: a factor $d/\ln d$ later than $u$.

In online SGD these population times are multiplied by $d/\delta$ samples per unit time, with $\delta$ limited by noise and renormalization (7.2.B). That gives about $d\log d$ to $d\log^2d$ for $u$, and for $v$ when correlated, versus $d^{\,3}$ for $v$ alone.
"""
    ),
    markdown(
        r"""
**Demo 7.2.E — The staircase in a two-layer ReLU network.**
Does a small two-layer ReLU network, trained by online SGD on the mixed cumulant model
in $d=16$, find the mean, covariance and cumulant directions in turn, and does
correlating the latents $\lambda,\nu$ shorten the last plateau? Test accuracy is measured
on the full data, on a "mean only" set ($\beta_u=\beta_v=0$), and on the Gaussian clone.
"""
    ),
    code(
        r"""
# Two-layer ReLU net on the mixed cumulant model (d = 16), online minibatch SGD.
# Test sets: full data, a "mean only" set (beta_u = beta_v = 0), and the Gaussian clone.
def mcm_run(correlated, d=16, width=64, steps=12000, B=32, lr=0.5, seed=0):
    spikes = orthonormal_spikes(d)
    gen = torch.Generator().manual_seed(seed + 10)
    tests = {"full data": mcm_sample(10000, d, spikes, correlated=correlated, gen=gen),
             "mean only": mcm_sample(10000, d, spikes, beta_u=0, beta_v=0, gen=gen),
             "Gaussian clone": mcm_sample(10000, d, spikes, correlated=correlated, clone=True, gen=gen)}
    X, Y = mcm_sample(steps * B, d, spikes, correlated=correlated, gen=gen)
    X, Y = X.view(steps, B, d), Y.view(steps, B)
    torch.manual_seed(seed)
    W = (torch.randn(width, d) / np.sqrt(d)).requires_grad_()
    b = torch.zeros(width).requires_grad_()
    a = (torch.randn(width) / np.sqrt(width)).requires_grad_()
    net = lambda x: torch.relu(x @ W.T + b) @ a
    record = sorted(set(np.unique(np.geomspace(1, steps, 40).astype(int)).tolist()))
    hist = {k: [] for k in tests}
    overlap = []
    for t in range(steps + 1):
        if t in record:
            with torch.no_grad():
                for k, (xt, yt) in tests.items():
                    hist[k].append((net(xt) * yt > 0).float().mean().item())
                overlap.append(((W @ spikes[2]).abs() / W.norm(dim=1)).max().item())
        if t == steps:
            break
        loss = torch.nn.functional.softplus(-Y[t] * net(X[t])).mean()   # logistic loss
        gW, gb, ga = torch.autograd.grad(loss, (W, b, a))
        with torch.no_grad():
            W -= lr * gW; b -= lr * gb; a -= lr * ga
    return record, hist, overlap

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, corr in zip(axes, [False, True]):
    ts_, hist, ov = mcm_run(corr)
    for k, c in zip(hist, ["C3", "C0", "C1"]):
        ax.semilogx(ts_, hist[k], color=c, label=k)
    ax.semilogx(ts_, ov, color="gray", lw=1, ls="--", label=r"max$_j|\cos(w_j,v)|$")
    if not corr:
        for level in (0.691, 0.767, 0.819):
            ax.axhline(level, color="gray", lw=0.7, ls=":")
    ax.set(xlabel="SGD step (batch 32, fresh samples)",
           title=("correlated" if corr else "independent") + " latents")
axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
*Left, independent latents.* Dotted lines: Bayes accuracies from the means (0.691), means
and covariances (0.767), and everything (0.819). The mean and covariance stages pass within
~20 steps, too fast to separate here. Then comes a long plateau near 0.75 on which full-data
and clone accuracies coincide: the network uses nothing the clone lacks. Near step $10^3$
the largest hidden-unit overlap with $v$ (dashed) jumps toward 1; full-data accuracy rises
to about 0.79 and the clone's falls. The "mean only" set stays below 0.691 because the
network also responds to variation along $u$ and $v$, which carries no class information
there.

*Right, correlated latents.* The overlap with $v$ rises from about step 100, and full data
separates from the clone within a few hundred steps. (The clone now carries some
information about $v$, 7.2.A.3.) With 64 units in $d=16$ some unit starts with a sizable
overlap by chance, so the dashed curves start near 0.6. At this size plateau lengths do not
test the exponents (7.2.C does, for one neuron); the demo shows the staircase.
"""
    ),
    # ================================================================ 7.3 (spectral bias)
    markdown(
        r"""
<a id="td27-3"></a>

## 7.3 Spectral bias: kernel gradient flow, mode by mode

Networks fitting a target made of several frequencies fit the low ones first, even when
the high ones have larger amplitude: the *spectral bias*, or *frequency principle*. For a
model linear in its parameters (a kernel method, random features, or a network in its
frozen-kernel, NTK, regime) trained on square loss, the order is exact: each kernel
eigenfunction is fitted at a rate equal to its eigenvalue.

**Papers:** Basri, Jacobs, Kasten & Kritchman, *The convergence rate of neural networks
for learned functions of different frequencies*; Rahaman et al., *On the spectral bias of
neural networks* (see [Readings](#td27-readings)).

*Core idea.* With fixed features $\phi(x)$ and kernel $K(x,x')=\phi(x)\cdot\phi(x')$,
gradient flow on square loss makes the residual decay as $e^{-\lambda_kt}$ along the
$k$-th eigenfunction of the kernel, where $\lambda_k$ is its eigenvalue under the data
distribution. So which functions count as "simple" is set by the kernel together with the
data distribution. For inputs uniform on a circle, the eigenfunctions are Fourier modes and
the eigenvalues are the kernel's Fourier coefficients. For a two-layer ReLU network's NTK
with biases they decay like $\ell^{-2}$, so frequency $\ell$ takes a time $\propto\ell^2$;
without biases, odd frequencies $\ge3$ cannot be fitted at all.

<a id="td27-3-walkthrough"></a>

**Derivations and walkthrough.** Background elsewhere: the Physics of Deep Learning day's
[Notebook 1, §3]({PHYS}/01_fixed_features_kernel_learning_solutions.ipynb) (kernel gradient
flow as spectral filtering; if you have done it, skim to *On the circle*) and its
[Notebook 2]({PHYS}/02_ntk_limit_kernel_learners_solutions.ipynb) (the infinite-width NTK).

*Notation.* A fixed-feature model $f(x)=\theta\cdot\phi(x)$ with kernel
$K(x,x')=\phi(x)\cdot\phi(x')$, trained on $(x_i,y_i)_{i\le n}$ with
$L=\frac1{2n}\sum_i(f(x_i)-y_i)^2$; residuals $r_i=f(x_i)-y_i$. On the circle,
$x=(\cos\vartheta,\sin\vartheta)$ with $\vartheta$ uniform.

**Mode by mode.** Gradient flow gives $\dot f(x)=-\frac1n\sum_iK(x,x_i)\,r_i$, so on the
training set $r(t)=e^{-\mathbf Kt/n}r(0)$, with Gram matrix $\mathbf K_{ij}=K(x_i,x_j)$. The
residual along an eigenvector of $\mathbf K/n$ with eigenvalue $\hat\lambda_k$ decays as
$e^{-\hat\lambda_kt}$, and a component with $\hat\lambda_k=0$ never moves (Exercise 7.3.A).
As $n\to\infty$ these eigenvalues approach those of the integral operator
$(T_Kg)(x)=\mathbb E_{x'}[K(x,x')g(x')]$, and the population flow is
$\partial_tf=-T_K(f-f^*)$. The shallow linear model of 7.1.A is the case $\phi(x)=x$, where
$T_K$ acts as the input covariance $\Sigma_{xx}$.

**On the circle.** For $K$ depending only on $\vartheta-\vartheta'$, $T_K$ is a
convolution. Its eigenfunctions are the Fourier modes $\cos\ell\vartheta$ and
$\sin\ell\vartheta$, and its eigenvalues are the Fourier coefficients of the kernel
(Exercise 7.3.B). Rotation invariance alone does not order them; that depends on the
kernel. Two cases worth knowing:

- **The von Mises kernel** $e^{\kappa(\cos(\vartheta-\vartheta')-1)}$, whose width is set by $\kappa>0$ (here $\kappa$ is not a cumulant). Its eigenvalues decay faster than exponentially in $\ell$. (Smoothness alone does not fix the order: $1+\cos8\varphi$ is a smooth kernel that fits only $\ell=0$ and $\ell=8$.)
- **The NTK of a two-layer ReLU network** with only the first layer trained (Basri et al.). Write the kernel as a function of the angle $\varphi=\vartheta-\vartheta'$ between inputs, $\sum_\ell a_\ell\cos\ell\varphi$. Without biases, its cosine coefficients are $a_0=\tfrac1{\pi^2}$, $a_1=\tfrac14$, $a_\ell=\tfrac{2(\ell^2+1)}{\pi^2(\ell^2-1)^2}$ for even $\ell\ge2$, and **zero for odd $\ell\ge3$**. (The eigenvalues are $a_0$ and $a_\ell/2$ for $\ell\ge1$.) So a bias-free shallow ReLU network cannot fit $\cos3\vartheta$ at all. With biases, every frequency is present and the eigenvalues decay like $\ell^{-2}$, so the time to learn frequency $\ell$ grows like $\ell^2$. Basri et al. measured convergence times growing roughly as $\ell^2$ for shallow and deep networks.

**The link to §7.1–7.2.** On the circle, $\cos\ell\vartheta=\mathrm{Ch}_\ell(\cos\vartheta)$
(Chebyshev polynomial of degree $\ell$) is a polynomial of degree $\ell$ in the input
$x=(\cos\vartheta,\sin\vartheta)$. So fitting frequency $\ell$ needs features of polynomial
degree at least $\ell$: the same grading by degree that ordered moments in 7.1.B and set the
information exponent (a Hermite degree) in §7.2. A dot-product kernel
$\sum_kb_k(x\cdot x')^k$ is graded the same way, since $(\cos\varphi)^k$ contains only
frequencies $\le k$ of the parity of $k$. The zeros of the bias-free ReLU NTK at odd
$\ell\ge3$ are 7.1.B.3's parity argument again: $\mathrm{ReLU}(z)=\tfrac z2+\tfrac{|z|}2$,
so the odd part of a bias-free ReLU network is linear in $x$, and its only odd frequency is
$\ell=1$.

*Scope.* "Rate = eigenvalue" is exact for a fixed kernel. A network's empirical NTK
($\nabla_\theta f(x)\cdot\nabla_\theta f(x')$ at the current weights) moves during training,
and the prediction holds only while it moves little (demo 7.3.D). For non-uniform data or
other domains, the order is by eigenvalue of $T_K$ under the data distribution, whatever the
eigenfunctions look like.

**Exercise 7.3.A — Mode-by-mode solution.** In the notation above:

1. Derive $\dot f(x)=-\frac1n\sum_iK(x,x_i)r_i$ and solve the training-set dynamics in the eigenbasis of $\mathbf K$.
2. Starting from $\theta=0$ (so $f\equiv0$), find $f_t(x)$ at an arbitrary point $x$, not only on the training set. With $\mathbf K/n=\sum_k\hat\lambda_kv_kv_k^\top$ and $K(x,X)$ the row vector $(K(x,x_i))_i$, show that
$$
f_t(x)=\sum_{k:\,\hat\lambda_k>0}K(x,X)\,v_k\;\frac{1-e^{-\hat\lambda_kt}}{n\hat\lambda_k}\;(v_k\cdot y).
$$
What is the limit $t\to\infty$?
3. Redo 1 for GD with step $\eta$. What limits $\eta$, and for $\eta\le1/\hat\lambda_{\max}$, how many steps does the slowest positive mode need?

<details><summary>Hint</summary>

For 2, write $\theta_t=\sum_i\alpha_i(t)\phi(x_i)$: gradient flow from $\theta=0$ stays in the span of the training features. Then $\mathbf K\alpha_t=f_t(X)=y+r_t$.
</details>
""".replace("{PHYS}", PHYS)
    ),
    answer(
        r"""
**1.** $\nabla_\theta L=\frac1n\sum_ir_i\phi(x_i)$, so $\dot\theta=-\frac1n\sum_ir_i\phi(x_i)$, and by the chain rule $\dot f(x)=\phi(x)\cdot\dot\theta=-\frac1n\sum_iK(x,x_i)r_i$. At $x=x_j$, $\dot r_j=\dot f(x_j)=-\frac1n\sum_i\mathbf K_{ji}r_i$, since the targets are constant. With $\mathbf K/n=\sum_k\hat\lambda_kv_kv_k^\top$,

$$
r(t)=e^{-\mathbf Kt/n}r(0)=\sum_ke^{-\hat\lambda_kt}\,(v_k\cdot r(0))\,v_k .
$$

Each eigen-direction of the residual decays independently at rate $\hat\lambda_k$. Directions in the null space of $\mathbf K$ do not move.

**2.** From $\theta=0$, $\dot\theta$ is always a combination of the $\phi(x_i)$, so $\theta_t=\sum_i\alpha_i(t)\phi(x_i)$ and $f_t(x)=\sum_i\alpha_iK(x,x_i)$. On the training set $f_t=\mathbf K\alpha_t=y+r_t=y-e^{-\mathbf Kt/n}y$, using $r_0=-y$. Solving on the range of $\mathbf K$,

$$
f_t(x)=\sum_{k:\,\hat\lambda_k>0} K(x,X)\,v_k\;\frac{1-e^{-\hat\lambda_kt}}{n\hat\lambda_k}\;(v_k\cdot y),
$$

where $K(x,X)$ is the row vector $(K(x,x_i))_i$. The filter factor $(1-e^{-\hat\lambda t})/\hat\lambda$ is $\approx t$ for $\hat\lambda t\ll1$ (the mode has barely started) and $\approx1/\hat\lambda$ for $\hat\lambda t\gg1$ (fitted). As $t\to\infty$ this is the minimum-norm least-squares solution. On the training set it equals $y$ minus the projection of $y$ onto the null space of $\mathbf K$, so it interpolates $y$ exactly when $y$ lies in the range of $\mathbf K$, and every $y$ when $\mathbf K\succ0$. At finite $t$ it is a spectral filter, which is why early stopping behaves like ridge regularization. The code below uses this formula.

**3.** GD: $\theta_{s+1}=\theta_s-\eta\nabla L$ gives $r_{s+1}=(I-\eta\mathbf K/n)r_s$, so $r_s=\sum_k(1-\eta\hat\lambda_k)^s(v_k\cdot r_0)v_k$. Mode $k$ shrinks by the factor $|1-\eta\hat\lambda_k|$ per step, so all positive modes decay iff $0<\eta<2/\hat\lambda_{\max}$; null-space components stay. For $\eta\le1/\hat\lambda_{\max}$ every factor is $1-\eta\hat\lambda_k\ge0$, and the slowest is the smallest positive eigenvalue: it needs about $1/(\eta\hat\lambda_{\min})\ge\hat\lambda_{\max}/\hat\lambda_{\min}$ steps per $e$-fold. (Closer to $2/\hat\lambda_{\max}$ the top mode itself oscillates and decays slowly.) The fastest mode limits the step size, and the slowest then sets the training time: the spread of the kernel spectrum is the spread of learning times.
"""
    ),
    markdown(
        r"""
**Exercise 7.3.B — Eigenvalues on the circle.**
Each component of the target is learned at a rate equal to an eigenvalue of
$(T_Kg)(\vartheta)=\mathbb E_{\vartheta'}[K(\vartheta,\vartheta')g(\vartheta')]$. Compute
these eigenvalues for kernels that depend only on $\vartheta-\vartheta'$.

1. For $\vartheta$ uniform on $[0,2\pi)$ and $K(\vartheta,\vartheta')=k(\vartheta-\vartheta')$ with $k$ even, show $T_K\cos\ell\vartheta=\hat k_\ell\cos\ell\vartheta$ and likewise for $\sin\ell\vartheta$, with $\hat k_\ell=\frac1{2\pi}\int_{-\pi}^{\pi}k(\varphi)\cos(\ell\varphi)\,d\varphi$.
2. For the von Mises kernel $k(\varphi)=e^{\kappa(\cos\varphi-1)}$ ($\kappa>0$ sets its width), show $\hat k_\ell=e^{-\kappa}I_\ell(\kappa)$, where $I_\ell$ is a modified Bessel function. Find its behaviour for large $\ell$. How does the learning time $1/\hat k_\ell$ grow with $\ell$, and what does a large $\kappa$ (a narrow kernel) change?
3. For the target $\sin\vartheta+\sin4\vartheta+\sin8\vartheta$, roughly how much later is the $\ell=8$ component fitted than the $\ell=1$ component under the von Mises kernel with $\kappa=4$? Compare with an ideal $\ell^{-2}$ spectrum (the asymptotic decay of the ReLU NTK with biases).

<details><summary>Hint</summary>

Jacobi–Anger: $e^{\kappa\cos\varphi}=I_0(\kappa)+2\sum_{\ell\ge1}I_\ell(\kappa)\cos\ell\varphi$, and $I_\ell(\kappa)=\sum_{j\ge0}\frac{(\kappa/2)^{2j+\ell}}{j!\,(j+\ell)!}$.
</details>
"""
    ),
    answer(
        r"""
**1.** With the uniform probability measure $d\vartheta'/2\pi$,

$$
(T_Ke^{i\ell\cdot})(\vartheta)=\frac1{2\pi}\int k(\vartheta-\vartheta')e^{i\ell\vartheta'}d\vartheta'
=e^{i\ell\vartheta}\,\frac1{2\pi}\int k(\varphi)e^{-i\ell\varphi}d\varphi
=\hat k_\ell\,e^{i\ell\vartheta},
$$

substituting $\varphi=\vartheta-\vartheta'$ and using periodicity. Since $k$ is even, $\hat k_\ell=\hat k_{-\ell}$ is real and equal to the cosine integral in the statement. $\cos\ell\vartheta$ and $\sin\ell\vartheta$ are combinations of $e^{\pm i\ell\vartheta}$ with the same eigenvalue. (Convention check: if $k(\varphi)=\sum_\ell a_\ell\cos\ell\varphi$, then $\hat k_0=a_0$ and $\hat k_\ell=a_\ell/2$ for $\ell\ge1$. Basri et al. quote the coefficients $a_\ell$.)

**2.** By Jacobi–Anger, $k(\varphi)=e^{-\kappa}\big[I_0(\kappa)+2\sum_{\ell\ge1}I_\ell(\kappa)\cos\ell\varphi\big]$, so $\hat k_\ell=e^{-\kappa}I_\ell(\kappa)$ for all $\ell\ge0$. For $\ell\gg\kappa^2/4$ the $j=0$ term of the series dominates, and

$$
\hat k_\ell\approx e^{-\kappa}\frac{(\kappa/2)^\ell}{\ell!},\qquad
t_\ell\sim\frac1{\hat k_\ell}\approx e^{\kappa}\,\ell!\,(2/\kappa)^\ell ,
$$

which grows faster than exponentially. For large $\kappa$, the kernel has angular width $\sim\kappa^{-1/2}$. For $\ell=O(\sqrt\kappa)$, the standard asymptotic $I_\ell(\kappa)\approx e^{\kappa}e^{-\ell^2/2\kappa}/\sqrt{2\pi\kappa}$ gives $\hat k_\ell\approx e^{-\ell^2/2\kappa}/\sqrt{2\pi\kappa}$: roughly flat up to $\ell\sim\sqrt\kappa$, falling off beyond. A narrower kernel fits more frequencies at comparable speed, at the price of a smaller overall eigenvalue scale.

**3.** At $\kappa=4$: $\hat k_1=0.179$, $\hat k_4=0.0259$, $\hat k_8=1.80\times10^{-4}$, so $t_8/t_1\approx1000$. Under an ideal $\ell^{-2}$ spectrum, $t_8/t_1=64$. (For the ReLU NTK with bias of 7.3.C, the actual coefficients give $a_1/a_8\approx200$, because the $\ell^{-2}$ law is only asymptotic.) Both kernels fit low frequencies first; how much earlier depends on the kernel.
"""
    ),
    code(
        r"""
# Kernel gradient flow on the circle: residual per frequency vs the prediction exp(-k_l t).
from scipy.special import ive          # ive(l, kappa) = exp(-kappa) I_l(kappa)

n, kappa, freqs = 256, 4.0, [1, 4, 8]
theta = 2 * np.pi * np.arange(n) / n                     # equispaced training inputs
kern = lambda a, b: np.exp(kappa * (np.cos(a[:, None] - b[None, :]) - 1))
target = lambda t: sum(np.sin(l * t) for l in freqs)
y = target(theta)

lam_hat, V = np.linalg.eigh(kern(theta, theta) / n)      # eigenpairs of K/n
lam_hat = np.clip(lam_hat, 0, None)
pos = lam_hat > 1e-13                                    # 7.3.A.2 sums over the positive spectrum
grid = 2 * np.pi * np.arange(1024) / 1024
K_grid = kern(grid, theta) / n
pred_rate = ive(np.array(freqs), kappa)                  # population eigenvalues k_l

def f_at(t):                                             # 7.3.A.2
    filt = -np.expm1(-lam_hat[pos] * t) / lam_hat[pos]
    return K_grid @ (V[:, pos] @ (filt * (V[:, pos].T @ y)))

ts = np.geomspace(0.3, 3e5, 80)
amps = np.array([np.abs(np.fft.rfft(f_at(t) - target(grid)))[freqs] * 2 / len(grid) for t in ts])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j, l in enumerate(freqs):
    ax[0].semilogx(ts, amps[:, j], color=f"C{j}", label=f"frequency {l}")
    ax[0].semilogx(ts, np.exp(-pred_rate[j] * ts), color="k", ls=":", lw=1)
ax[0].set(xlabel="time t", ylabel="residual amplitude",
          title=r"kernel gradient flow (dotted: $e^{-\hat k_\ell t}$)")
ax[0].legend(fontsize=8)
for t, c in zip([3, 100, 3000], ["C3", "C4", "C5"]):
    ax[1].plot(grid, f_at(t), color=c, label=f"t = {t}")
ax[1].plot(grid, target(grid), "k", lw=0.8, label="target")
ax[1].set(xlabel=r"$\vartheta$", title="fitted function: low frequencies first")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("population eigenvalues k_l:", dict(zip(freqs, np.round(pred_rate, 6))))
"""
    ),
    markdown(
        r"""
With equispaced inputs the Gram matrix is circulant, so its eigenvectors are exactly
Fourier modes and the curves sit on the predictions (aliases $\hat k_{\ell+jn}$ are
negligible). Try random inputs (`theta = rng.uniform(0, 2*np.pi, n)`): each frequency still
follows its exponential at first, but some unfitted high-frequency residual leaks into the
low-frequency coefficients, leaving a floor of a few per cent (shrinking with $n$).

**Exercise 7.3.C — Code: the ReLU NTK spectrum on the circle.**
Take Basri et al.'s two-layer ReLU network $f(x)=\frac1{\sqrt N}\sum_{r=1}^Na_r\,\mathrm{ReLU}(w_r\cdot x)$, with $x=(\cos\vartheta,\sin\vartheta)$, Gaussian $w_r$, and only $W$ trained ($a_r=\pm1$ fixed). Its first-layer NTK, $\frac1N\sum_r x\cdot x'\,\mathbf 1[w_r\cdot x>0]\,\mathbf 1[w_r\cdot x'>0]$, has infinite-width limit

$$
K^\infty(\vartheta)=\frac{1}{2\pi}\,\cos\vartheta\,\big(\pi-|\vartheta|\big),\qquad \vartheta\in[-\pi,\pi].
$$

Add a bias drawn like the other weights, by using $\bar x=(x,1)/\sqrt2$ in place of $x$. The kernel becomes $\frac1{2\pi}u(\pi-\arccos u)$ with $u=\bar x\cdot\bar x'=\frac{1+\cos\vartheta}2$. (Basri et al.'s eq. 13 uses a zero-initialized bias instead; its constants differ, and it also decays as $\ell^{-2}$.)

1. Compute the cosine coefficients $a_\ell=\frac1{z_\ell}\int_{-\pi}^{\pi}K^\infty(\vartheta)\cos\ell\vartheta\,d\vartheta$ ($z_0=2\pi$, $z_\ell=\pi$) numerically for $\ell\le20$, and compare with the closed form in the walkthrough above.
2. Do the same for the bias kernel. Which coefficients become nonzero? Fit the decay exponent of $a_\ell$ for $\ell\ge4$.
3. Check the first formula by Monte Carlo. Sample $N=20000$ random $w_r$, build the finite-width kernel on a grid of angles, and compare its cosine coefficients for $\ell\le6$.
"""
    ),
    code_answer(
        r"""
from scipy.integrate import quad

K_nobias = lambda t: np.cos(t) * (np.pi - abs(t)) / (2 * np.pi)
def K_bias(t):
    u = (1 + np.cos(t)) / 2
    return u * (np.pi - np.arccos(np.clip(u, -1, 1))) / (2 * np.pi)

def cos_coef(F, l):
    # Your code: a_l = (1/z_l) * integral_{-pi}^{pi} F(t) cos(l t) dt, with z_0 = 2 pi, z_l = pi.
    ...

def basri(l):
    # Your code: the closed form of section 7.3 (no bias).
    ...

# Your code: compare for l <= 20; repeat for K_bias and fit a power law for l >= 4;
# then the Monte Carlo check with N = 20000 Gaussian first-layer weights.
""",
        r"""
from scipy.integrate import quad

K_nobias = lambda t: np.cos(t) * (np.pi - abs(t)) / (2 * np.pi)
def K_bias(t):
    u = (1 + np.cos(t)) / 2
    return u * (np.pi - np.arccos(np.clip(u, -1, 1))) / (2 * np.pi)

def cos_coef(F, l):
    z = 2 * np.pi if l == 0 else np.pi
    return quad(lambda t: F(t) * np.cos(l * t), -np.pi, np.pi, points=[0], limit=200)[0] / z

def basri(l):
    if l == 0: return 1 / np.pi**2
    if l == 1: return 1 / 4
    return 2 * (l * l + 1) / (np.pi**2 * (l * l - 1) ** 2) if l % 2 == 0 else 0.0

ls = np.arange(21)
a_nb = np.array([cos_coef(K_nobias, l) for l in ls])
a_b = np.array([cos_coef(K_bias, l) for l in ls])
print("max |numerical - closed form| (no bias):", np.abs(a_nb - [basri(l) for l in ls]).max())
print("odd coefficients l>=3, no bias:  ", np.round(a_nb[3::2][:5], 12))
print("odd coefficients l>=3, with bias:", np.round(a_b[3::2][:5], 5))
fit = np.polyfit(np.log(ls[4:]), np.log(a_b[4:]), 1)
print(f"bias kernel: a_l ~ l^{fit[0]:.2f} for l >= 4")

# Monte Carlo: finite-width first-layer kernel on an equispaced grid of angles.
W_mc = rng.normal(size=(20000, 2))
ang = 2 * np.pi * np.arange(256) / 256
X_ang = np.c_[np.cos(ang), np.sin(ang)]
act = (X_ang @ W_mc.T > 0).astype(float)
row = (X_ang @ X_ang[0]) * (act @ act[0]) / W_mc.shape[0]   # kernel against angle 0
c_hat = np.fft.rfft(row).real / len(ang)
a_mc = np.r_[c_hat[0], 2 * c_hat[1:7]]                      # cosine coefficients for l = 0..6
print("Monte Carlo a_l, l<=6:", np.round(a_mc, 4))
print("closed form    l<=6:", np.round([basri(l) for l in range(7)], 4))

plt.figure(figsize=(5, 3.4))
plt.loglog(ls[1:], np.maximum(a_nb[1:], 1e-12), "o", label="no bias")
plt.loglog(ls[1:], a_b[1:], "s", mfc="none", label="with bias")
plt.loglog(ls[2:], 1 / ls[2:] ** 2 / 4, "k--", lw=0.8, label=r"$\propto \ell^{-2}$")
plt.ylim(1e-5, 1); plt.xlabel(r"frequency $\ell$"); plt.ylabel(r"cosine coefficient $a_\ell$")
plt.legend(fontsize=8); plt.title("two-layer ReLU NTK on the circle"); plt.tight_layout(); plt.show()
""",
    ),
    markdown(
        r"""
The closed form matches to quadrature precision, and the odd coefficients vanish
without a bias. With a bias every coefficient is positive, falling off like $\ell^{-2}$.

**Demo 7.3.D — A trained ReLU network against the kernel prediction.**
Does a finite two-layer ReLU network, trained by GD on the target
$\sin\vartheta+\sin4\vartheta+\sin8\vartheta$ of the kernel demo above, fit each frequency
when its infinite-width kernel (7.3.C, with biases) predicts? Two output scales compare a
run near the lazy, frozen-kernel limit (scale 10) with one whose features move (scale 1).
"""
    ),
    code(
        r"""
# A two-layer ReLU network in Basri et al.'s setting (first layer and biases trained,
# readout fixed), fitted by full-batch GD to the target of the kernel demo.
# `scale` multiplies the output; the step is divided by scale^2, so the function-space
# dynamics are the same at leading order but the weights move less (the lazy limit).
n_tr, pairs, steps, fq = 128, 1000, 2000, np.array([1, 4, 8])
th = 2 * np.pi * np.arange(n_tr) / n_tr
y_tr = sum(np.sin(l * th) for l in fq)
Xb = np.c_[np.cos(th), np.sin(th), np.ones(n_tr)].astype(np.float32)   # the bias is w[2]
W_init = np.random.default_rng(7).standard_normal((pairs, 3)).astype(np.float32)
W_init = np.r_[W_init, W_init]                     # twin units with opposite readouts,
a_out = np.r_[np.ones(pairs), -np.ones(pairs)].astype(np.float32)   # so f starts at 0
N = 2 * pairs
from scipy.integrate import quad
def K_bias_(t):                                    # 7.3.C's bias kernel; the eNTK here averages to 2 K_bias_
    u = (1 + np.cos(t)) / 2
    return u * (np.pi - np.arccos(np.clip(u, -1, 1))) / (2 * np.pi)
a_l = {l: quad(lambda t: K_bias_(t) * np.cos(l * t), -np.pi, np.pi, points=[0])[0] / np.pi for l in fq}

def first_layer_ntk(W, scale):                     # K/n for the trained first layer
    m = (Xb @ W.T > 0).astype(np.float32)
    return scale**2 * (Xb @ Xb.T) * (m @ m.T) / N / n_tr

def train_basri(scale):
    W = W_init.copy()
    K0 = first_layer_ntk(W, scale)
    eta = 1.0 / np.linalg.eigvalsh(K0.astype(float))[-1]
    amps = []
    for s in range(steps + 1):
        pre = Xb @ W.T
        f = scale * (np.maximum(pre, 0) @ a_out) / np.sqrt(N)
        amps.append(np.abs(np.fft.rfft(f - y_tr))[fq] * 2 / n_tr)
        if s == steps:
            break
        g = scale * (((pre > 0) * a_out) * (f - y_tr)[:, None]).T @ Xb / (np.sqrt(N) * n_tr)
        W -= (eta * g).astype(np.float32)
    K1 = first_layer_ntk(W, scale)
    dK = np.linalg.norm(K1 - K0) / np.linalg.norm(K0)
    sl = np.sin(fq[-1] * th)                       # Rayleigh quotient on the slowest mode
    rq = [eta * sl @ K @ sl / (sl @ sl) for K in (K0, K1)]
    return np.array(amps), eta, dK, rq

ts = np.arange(steps + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for scale, ls_ in [(1.0, "-"), (10.0, "--")]:
    A, eta, dK, rq = train_basri(scale)
    rate_unit = eta * scale**2                       # function-space step; the same for both runs
    t_fit = [int(ts[np.argmax(A[:, j] < np.exp(-1))]) if (A[:, j] < np.exp(-1)).any() else np.nan
             for j in range(len(fq))]
    print(f"scale {scale:>4}: eNTK change {dK:.3f}; steps to residual 1/e at l = {fq.tolist()}: {t_fit}; "
          f"eta*K on sin(8 theta): {rq[0]:.5f} at init, {rq[1]:.5f} at the end")
    for j, l in enumerate(fq):
        ax[0].semilogx(ts + 1, A[:, j], color=f"C{j}", ls=ls_, label=rf"$\ell$ = {l}" if scale == 1 else None)
    ax[1].loglog(fq, t_fit, "o" if scale == 1 else "s", mfc="none", label=f"network, scale {scale:g}")
t_pred = np.ceil(-1 / np.log(1 - np.array([rate_unit * a_l[l] for l in fq])))   # first GD step with (1 - eta a)^t < 1/e
print("prediction from the infinite-width kernel:", np.round(t_pred, 1).tolist())
for j, l in enumerate(fq):
    ax[0].semilogx(ts + 1, (1 - rate_unit * a_l[l]) ** ts, color="k", ls=":", lw=1)
ax[0].set(xlabel="GD step + 1", ylabel="residual amplitude",
          title="solid: scale 1, dashed: scale 10, dotted: 7.3.C")
ax[0].legend(fontsize=8)
ax[1].loglog(fq, t_pred, "k+", ms=10, label="infinite-width prediction")
ax[1].loglog(fq, t_pred[0] * fq**2.0, "k:", lw=0.8, label=r"$\propto\ell^2$")
ax[1].set(xlabel=r"frequency $\ell$", ylabel=r"steps until residual $<e^{-1}$")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
The network's kernel at initialization averages to $2K_{\rm bias}$ of 7.3.C, so the
prediction for frequency $\ell$ uses the eigenvalue $a_\ell$; the inverse eigenvalues are in
the ratio of about 1 : 46 : 200 for $\ell=1,4,8$ (the $\ell^{-2}$ law is only asymptotic).
The lazy run (output scale 10, step reduced 100-fold) fits frequency 1, then 4, then 8, at
the predicted times, and its eNTK barely moves. At scale 1, frequency 8 is still at about
half its initial amplitude after 2000 steps, and frequency 4 lags too. The printout shows
why: the eNTK changes by only a few per cent overall, but its Rayleigh quotient on
$\sin8\vartheta$ drops about five-fold (at scale 10, by about a fifth). Feature learning has
shrunk the kernel exactly on the slowest mode, so the frozen-kernel prediction is most
fragile for the modes it says are slow.
"""
    ),
]

raise SystemExit(run(CELLS, "td2_7_learned_first", "td2_7_learned_first"))
