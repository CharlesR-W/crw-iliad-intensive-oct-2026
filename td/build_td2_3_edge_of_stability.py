#!/usr/bin/env python3
"""Build NB2 §3, "The edge of stability" (Training Dynamics day, Iliad Intensive 2026).

Shape (readings first, 2026-10-14): a short pitch and a one-line route, then one
`## Readings` section (anchor td23-readings) with every paper of the notebook, grouped by
section, then the derivations and walkthrough: eight `##` sections, each with its
question, a short core-idea paragraph and a one-line pointer to its papers, then (from
the td23-N-walkthrough anchor) the prerequisites, derivations, bold-labelled exercises
with folded solutions, and code. Sections: 3.1 phenomenology, 3.2 self-stabilization,
3.3 central flow, 3.4 Rod Flow and principal flow, 3.5 adaptive optimizers, 3.6 edge of
stochastic stability, 3.7 catapult, 3.8 warmup and WSD.

Content map: TD_SPECS_v1.1.md NB2 §3 (items 3.1-3.8), under td/TD_BUILD_DECISIONS.md.
Formulas are checked by td/checks/td2_3_edge_of_stability_sympy.py; the interactive
figure by td/assets/interactive/checks/eos_toy_and_catapult_check.py. Review log:
reviews/td/td2_3_edge_of_stability.md.
"""
from nbkit_td import answer, code, code_answer, figure, markdown, reading, run

FIG_TOY = figure(
    "assets/interactive/eos_toy_and_catapult.html",
    "Panel 1 is this toy, with sliders for $\\eta$ and the initial sharpness and an optional warmup ramp (its noise uses a different random generator, so runs near a survival boundary can differ from the notebook's).",
)
FIG_CAT = figure(
    "assets/interactive/eos_toy_and_catapult.html",
    "Panel 2 iterates these maps, with sliders for $\\eta\\lambda_0$ and the width $n$.",
)

CELLS = [
    # ================================================================== title + pitch
    markdown(
        r"""
# NB2 §3 — The edge of stability

**Training Dynamics, Iliad Intensive 2026**

Gradient descent with step $\eta$ on a quadratic of curvature $S$ is stable only if $\eta S<2$. Neural networks trained with full-batch gradient descent push their sharpness $S$ (the top Hessian eigenvalue) *up* to $2/\eta$ and then stay there, at the *edge of stability* (EoS), while the loss keeps falling on average. Below: why, a flow for the averaged trajectory, and the same idea in Adam, SGD, the *catapult*, and warmup and decay schedules.

**Route.** The [Readings](#td23-readings) come first: every paper of the notebook, grouped by section. The sections after them are the derivations and walkthrough (exercises with worked solutions, and code: CPU, under a minute per cell), for whoever wants them. Short path: 3.1 → 3.2 → 3.3.

- [3.1 Phenomenology](#td23-1): sharpness rises to $2/\eta$ and stays; no small-step flow can do this.
- [3.2 Self-stabilization](#td23-2): the oscillation pushes the sharpness back down; solved in a two-parameter toy.
- [3.3 The central flow](#td23-3): an ODE for the time-averaged iterate, and the loss it predicts. Companion site: [centralflows.github.io](https://centralflows.github.io/).
- [3.4 Rod Flow and principal flow](#td23-4): two other continuous-time models.
- [3.5 Adaptive optimizers](#td23-5): Adam's edge is a preconditioned sharpness at $38/\eta$.
- [3.6 The edge of stochastic stability](#td23-6): SGD's edge is *batch sharpness*.
- [3.7 The catapult](#td23-7): a step too large from the start, solved exactly in a wide two-layer model.
- [3.8 Warmup and WSD](#td23-8): why ramping $\eta$ up helps, and what the decay phase reveals.
"""
    ),
    code(
        r"""
import math, time
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)   # tiny models: one thread is fastest and avoids contention
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

def running_mean(a, m):
    # mean over a window of m samples; returns (centre indices, means), no edge padding
    return np.arange(m // 2, len(a) - m + 1 + m // 2), np.convolve(a, np.ones(m) / m, mode="valid")
"""
    ),
    # ================================================================== readings
    markdown(
        r"""
<a id="td23-readings"></a>

## Readings

**Phenomenology ([§3.1](#td23-1)).** The notebook's key paper.
"""
        + reading("Gradient descent on neural networks typically occurs at the edge of stability", "https://arxiv.org/abs/2103.00065",
                  "Cohen, Kaur, Li, Kolter, Talwalkar (ICLR 2021)",
                  "The key paper: progressive sharpening and the edge of stability across architectures, losses and datasets, with momentum thresholds and an appendix of caveats (cross-entropy, batch norm, ReLU).",
                  importance=5, length=3)
        + r"""
**Self-stabilization ([§3.2](#td23-2)).**
"""
        + reading("Self-stabilization: the implicit bias of gradient descent at the edge of stability", "https://arxiv.org/abs/2209.15594",
                  "Damian, Nichani, Lee (ICLR 2023)",
                  r"The mechanism, the four-stage cycle, and a theorem that gradient descent tracks projected gradient descent on $\min L$ s.t. $S\le2/\eta$ for a bounded number of steps. Appendix B has the toy of §3.2.",
                  importance=4, length=3)
        + r"""
**The central flow ([§3.3](#td23-3)).** Start with the companion site, then the paper.
"""
        + reading("Central flows (companion site)", "https://centralflows.github.io/",
                  "Cohen, Damian, Talwalkar, Kolter, Lee (2025)",
                  "The best introduction to the edge of stability and the central flow: three short illustrated parts on [gradient descent](https://centralflows.github.io/part1/), [scalar RMSProp](https://centralflows.github.io/part2/) and [RMSProp](https://centralflows.github.io/part3/). Read Part I before §3.3.",
                  importance=5, length=2)
        + reading("Understanding optimization in deep learning with central flows", "https://arxiv.org/abs/2410.24206",
                  "Cohen, Damian, Talwalkar, Kolter, Lee (ICLR 2025; arXiv v2 is the full version)",
                  r"The central flow for one and several unstable directions (the $\max(0,\cdot)$ form is in App. A.2.2), its loss prediction, and the RMSProp version, whose stationary preconditioner solves a convex program; extensive comparisons with real gradient descent.",
                  importance=4, length=4)
        + r"""
**Rod Flow and principal flow ([§3.4](#td23-4)).**
"""
        + reading("Rod Flow: a continuous-time model for gradient descent at the edge of stability", "https://arxiv.org/abs/2602.01480",
                  "Regis, Chewi (2026)",
                  "Exact midpoint dynamics of the rod, an explicit ODE needing only gradients and Hessian-vector products, the quadratic threshold and quartic self-stabilization, and the central flow as a Taylor approximation.",
                  importance=3, length=3)
        + reading("On a continuous time model of gradient descent dynamics and instability in deep learning", "https://arxiv.org/abs/2302.01952",
                  "Rosca, Wu, Qin, Dherin (TMLR 2023)",
                  "The principal flow: Exercise 3.1 for every Hessian eigendirection; explains the onset of instability, not the stabilization.",
                  importance=2, length=3)
        + r"""
Also: [A Rod Flow model for Adam at the edge of stability](https://arxiv.org/abs/2605.06821) (follow-up for momentum methods and Adam).

**Adaptive optimizers ([§3.5](#td23-5)).**
"""
        + reading("Adaptive gradient methods at the edge of stability", "https://arxiv.org/abs/2207.14484",
                  "Cohen, Ghorbani, Krishnan, Agarwal, et al. (2022 preprint; cite v2, which corrects Adam's preconditioner formula)",
                  r"Adam's preconditioned sharpness sits at $38/\eta$ while raw sharpness keeps rising; the preconditioner-feedback mechanism, and how minibatching changes the picture.",
                  importance=4, length=3)
        + r"""
**The edge of stochastic stability ([§3.6](#td23-6)).**
"""
        + reading("Edge of Stochastic Stability: revisiting the edge of stability for SGD", "https://arxiv.org/abs/2412.20553",
                  "Andreyev, Beneventano (arXiv v5, Dec 2025)",
                  r"Batch sharpness (their Definition 3) plateaus at $2/\eta$ for every batch size while $\lambda_{\max}$ settles lower; App. J compares other candidate quantities. Use v5: v1 used $\mathbb E_B\lambda_{\max}(H_B)$, and v2–v4 wrote the ratio of expectations in the definition while their experiments measured the expectation of the ratio.",
                  importance=4, length=3)
        + r"""
Also: [Momentum further constrains sharpness at the edge of stochastic stability](https://proceedings.mlr.press/v306/andreyev26a.html) (ICML 2026; with momentum $\beta$, batch sharpness plateaus at $2(1-\beta)/\eta$ for small batches and $2(1+\beta)/\eta$ for large ones); [SGD at the edge of stability: the stochastic sharpness gap](https://arxiv.org/abs/2604.21016) (2026; predicts a $\lambda_{\max}$ gap of roughly $1/b$, quantitatively untested).

**The catapult ([§3.7](#td23-7)).**
"""
        + reading("The large learning rate phase of deep learning: the catapult mechanism", "https://arxiv.org/abs/2003.02218",
                  "Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari (2020)",
                  r"The exact maps and three phases, the empirical $12/\lambda_0$ edge for ReLU networks, and evidence that the catapult phase often generalizes best.",
                  importance=4, length=3)
        + r"""
**Warmup and WSD ([§3.8](#td23-8)).**
"""
        + reading("Why warmup the learning rate? Underlying mechanisms and improvements", "https://arxiv.org/abs/2406.09405",
                  "Kalra, Barkeshli (NeurIPS 2024)",
                  "Warmup works by moving the network to flatter regions that tolerate the target step; two regimes set by initialization; a catapult-based choice of the initial step; a second-moment initialization for Adam.",
                  importance=4, length=3)
        + reading("Understanding warmup-stable-decay learning rates: a river valley loss landscape perspective", "https://arxiv.org/abs/2410.05192",
                  "Wen, Li, Wang, Hall, Liang, Ma (ICLR 2025)",
                  "The river-valley picture of WSD schedules: noise-driven oscillation across the valley hides progress along it until the decay phase; proposes WSD-S.",
                  importance=3, length=3)
    ),
    # ================================================================== 3.1
    markdown(
        r"""
<a id="td23-1"></a>

## 3.1 Phenomenology

What happens to the sharpness $S=\lambda_{\max}(H)$, the top eigenvalue of the loss Hessian $H$, during full-batch gradient descent with step $\eta$? It rises (*progressive sharpening*) to $2/\eta$, the quadratic stability limit, and hovers there while the loss keeps falling, non-monotonically.

**Paper:** Cohen et al. 2021, *Gradient descent on neural networks typically occurs at the edge of stability*, the notebook's key paper (see [Readings](#td23-readings); networks of several kinds, mostly on a 5,000-example CIFAR-10 subset).

Above $2/\eta$ a quadratic analysis predicts divergence along the top eigenvector $u$. Instead $S$ stays at or just above $2/\eta$ (further above with cross-entropy); with heavy-ball momentum $\beta$ the ceiling is $(2+2\beta)/\eta$. The sharpness *adapts to the learning rate*, which gradient flow, having no $\eta$ in it, cannot do; Exercise 3.1 shows its small-$\eta$ corrections cannot either. Why sharpening happens is a separate, partly open question, taken as given here (a tangent-kernel view: [`td2_6_curvature`, §6.2](td2_6_curvature.html#td26-2)).

<a id="td23-1-walkthrough"></a>

**Derivations and walkthrough.** *Prerequisites:* Gradient descent on $\tfrac12\lambda x^2$ is $x_{k+1}=(1-\eta\lambda)x_k$: it converges iff $0<\eta\lambda<2$, and for $\eta\lambda>2$ it flips sign every step and grows. On a positive-definite quadratic this holds along each Hessian eigenvector, so the constraint is $\eta S<2$. Time is $t=k\eta$ after $k$ steps, so gradient flow is $\dot\theta=-\nabla L$. A Hessian-vector product $Hv=\nabla_\theta\langle\nabla L,v\rangle$ costs about two gradients; *power iteration* $v\leftarrow Hv/\|Hv\|$ then estimates $S$ as $v^\top Hv$ (fine here, where the top eigenvalue is positive and dominant).

**The experiment.** A two-hidden-layer tanh MLP (width 64) fits 256 random $\pm1$ labels in $\mathbb R^8$ with MSE (random labels make sharpening fast), by full-batch gradient descent at three step sizes, with $S$ measured every $\Delta t=1.25$. In $t=k\eta$ all three runs follow the same gradient-flow curve until their own $2/\eta$. Each then overshoots (to about $1.5\times2/\eta$ at $\eta=0.2$), falls below with a loss spike, and settles into shrinking oscillations around $2/\eta$: the self-stabilization cycle of §3.2 at full size.
"""
    ),
    code(
        r"""
# A small MLP on 256 random +-1 labels; sharpness by power iteration on Hessian-vector products.
torch.manual_seed(0)
n_data, d_in, width = 256, 8, 64
X = torch.randn(n_data, d_in)
Y = torch.sign(torch.randn(n_data, 1))

def make_mlp(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1))

def mse(net):
    return 0.5 * ((net(X) - Y) ** 2).mean()

def sharpness(net, v=None, iters=20):
    # Power iteration with Hv = d/dtheta <grad L, v>. Returns (Rayleigh quotient, last vector).
    # Pass the previous vector back in as a warm start: then a few iterations per call suffice.
    params = list(net.parameters())
    grads = torch.autograd.grad(mse(net), params, create_graph=True)
    if v is None:
        v = [torch.randn_like(p) for p in params]
    for _ in range(iters):
        norm = torch.sqrt(sum((a * a).sum() for a in v))
        v = [a / norm for a in v]
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(Hv, v)).item()
        v = [a.detach() for a in Hv]
    return lam, v

def gd_step(net, eta):
    loss = mse(net)
    grads = torch.autograd.grad(loss, list(net.parameters()))
    with torch.no_grad():
        for p, g in zip(net.parameters(), grads):
            p -= eta * g
    return loss.item()

def run_gd(eta, t_max, dt_measure=1.25, seed=0):
    net = make_mlp(seed)
    every = max(1, round(dt_measure / eta))
    v, ts, Ss, losses = None, [], [], []
    for k in range(round(t_max / eta) + 1):
        if k % every == 0:
            lam, v = sharpness(net, v, iters=30 if v is None else 6)
            ts.append(k * eta); Ss.append(lam)
        losses.append(gd_step(net, eta))
    return np.array(ts), np.array(Ss), np.array(losses)

t0 = time.time()
runs = {eta: run_gd(eta, t_max=250) for eta in (0.05, 0.1, 0.2)}
print(f"trained 3 runs in {time.time() - t0:.1f} s")
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for (eta, (ts, Ss, losses)), c in zip(runs.items(), ["C0", "C1", "C2"]):
    ax[0].plot(ts, Ss, color=c, label=f"$\\eta$ = {eta}")
    ax[0].axhline(2 / eta, color=c, ls="--", lw=1)
    ax[1].semilogy(np.arange(len(losses)) * eta, losses, color=c, lw=0.8)
ax[0].set(xlabel="$t = k\\eta$", ylabel="sharpness $S$", title="sharpness (dashed: $2/\\eta$)")
ax[1].set(xlabel="$t = k\\eta$", ylabel="training loss", title="loss: non-monotone at EoS, decreasing overall")
ax[0].legend(); plt.tight_layout(); plt.show()
for eta, (ts, Ss, _) in runs.items():
    t_cross = ts[np.argmax(Ss >= 2 / eta)]
    late = Ss[ts > t_cross + 0.5 * (ts[-1] - t_cross)]          # second half of the time spent at EoS
    print(f"eta = {eta:4}: reaches 2/eta = {2/eta:4.0f} at t = {t_cross:5.1f}; peak S = {Ss.max():5.1f}; "
          f"S over the second half of EoS: mean {late.mean():.2f}, range [{late.min():.2f}, {late.max():.2f}]")
"""
    ),
    markdown(
        r"""
**Exercise 3.1 — The exact modified flow on a quadratic.** Why can't a small-step flow, however many correction terms it keeps, describe gradient descent at the edge? (If you did Exercise 1.2.c of [`td2_1_flows`](td2_1_flows.html#td21-s2), this is the same calculation; only the last paragraph of the solution is new.) Background: for small $\eta$, one gradient step is, to $O(\eta^3)$, a time-$\eta$ step of the *modified flow* $\dot\theta=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$ (implicit gradient regularization). Take gradient descent on $L(x)=\tfrac12\lambda x^2$ with $\lambda>0$, and $t=k\eta$.

1. For $0<\eta\lambda<1$, find the constant $a(\eta\lambda)$ such that the linear flow $\dot x=a\,x$, run for time $\eta$, reproduces one gradient step *exactly*.
2. Expand $a$ in powers of $\eta$. Identify the first two terms with gradient flow and with the implicit-gradient-regularization correction $-\tfrac\eta4\nabla\|\nabla L\|^2$. What is the radius of convergence of the series in $\eta\lambda$?
3. What happens to $a$ when $1<\eta\lambda<2$, and when $\eta\lambda>2$? Show that no real one-dimensional autonomous ODE $\dot x=F(x)$ can reproduce gradient descent for $\eta\lambda>1$, and say what this implies for any truncation of the modified-flow series at the edge of stability.

<details><summary>Hint</summary>

1. After $k$ steps $x_k=(1-\eta\lambda)^kx_0$; write this as an exponential in $t=k\eta$.
2. Use $\log(1-z)=-z-z^2/2-z^3/3-\dots$. For a quadratic, $\nabla L=\lambda x$ and $H\nabla L=\lambda^2x$.
3. Write $1-\eta\lambda=-|1-\eta\lambda|$ and take a complex logarithm. For the ODE: two solutions of a 1-D autonomous ODE cannot cross, so its time-$\eta$ map is an increasing function of the starting point. Is the gradient-descent map $x\mapsto(1-\eta\lambda)x$ increasing?
</details>
"""
    ),
    answer(
        r"""
**1.** $x_k=(1-\eta\lambda)^kx_0=\exp\!\big(k\log(1-\eta\lambda)\big)x_0=\exp\!\big(\tfrac t\eta\log(1-\eta\lambda)\big)x_0$. So the flow $\dot x=a\,x$ with

$$
a(\eta\lambda)=\frac1\eta\log(1-\eta\lambda)
$$

reproduces every iterate exactly at $t=k\eta$.

**2.** $a=\frac1\eta\big(-\eta\lambda-\tfrac12\eta^2\lambda^2-\tfrac13\eta^3\lambda^3-\dots\big)=-\lambda-\tfrac\eta2\lambda^2-\tfrac{\eta^2}3\lambda^3-\dots$ Hence

$$
\dot x=-\lambda x-\tfrac\eta2\lambda^2x+O(\eta^2)=-\nabla L-\tfrac\eta2H\nabla L+O(\eta^2).
$$

Since $\nabla\big(\tfrac\eta4\|\nabla L\|^2\big)=\tfrac\eta4\cdot2H\nabla L=\tfrac\eta2H\nabla L$, the first two terms are $-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$: gradient flow plus implicit gradient regularization. The series for $\log(1-z)$ converges for $|z|<1$, so the expansion converges only for $\eta\lambda<1$.

**3.** For $1<\eta\lambda<2$: $1-\eta\lambda\in(-1,0)$, and $\log(1-\eta\lambda)=\log|1-\eta\lambda|+i\pi$. The real part is negative (the magnitude still decays), but the imaginary part $i\pi/\eta$ rotates the solution by $\pi$ per step: the sign flip. For $\eta\lambda>2$: the real part $\frac1\eta\log(\eta\lambda-1)$ is *positive*, so the magnitude grows: the instability.

No real 1-D autonomous ODE can do this for $\eta\lambda>1$. Two solutions of $\dot x=F(x)$ cannot cross (uniqueness), so if $x_0<x_0'$ then $x(\eta)<x'(\eta)$: the time-$\eta$ map is increasing. The gradient-descent map $x\mapsto(1-\eta\lambda)x$ is decreasing when $\eta\lambda>1$. (The exact interpolating flow needs a second real dimension: the complex phase, or equivalently a period-2 structure. Rod Flow in §3.4 adds that dimension explicitly.)

So the networks, sitting at $\eta S\approx2$ without diverging, need something beyond the quadratic: the curvature changing along the oscillation must feed back. And any truncation of the modified-flow series is a smooth real flow, analytic in $\eta\lambda$ around $0$: it cannot represent the sign flip, and at $\eta S\approx2$ the series has already diverged. A flow for EoS must be built differently (§3.3, §3.4). Applied eigendirection by eigendirection, this exact flow is the *principal flow* of §3.4.
"""
    ),
    markdown(
        r"""
**Code exercise 3.1 — Is the oscillation really along $u$?** §3.2 assumes that at EoS the iterate oscillates with period 2 along the top Hessian eigenvector $u$. Check this in the MLP: train at $\eta=0.2$ for 1000 steps (into EoS), compute $u$ by power iteration, run 40 more steps, and look at the projections $\langle\theta_k-\bar\theta,u\rangle$, where $\bar\theta$ is the mean of those 40 iterates. Also report what fraction of each step $\theta_{k+1}-\theta_k$ (in squared norm) lies along $u$.

<details><summary>Hint</summary>

`torch.nn.utils.parameters_to_vector(net.parameters())` flattens the parameters in the same order as `list(net.parameters())`, so `torch.cat([t.flatten() for t in v])` turns the vector returned by `sharpness` into a flat $u$ (normalize it).
</details>
"""
    ),
    code_answer(
        r"""
# Code exercise 3.1: train at eta = 0.2 for 1000 steps, get u, then record 40 more iterates.
from torch.nn.utils import parameters_to_vector as p2v
# ...
""",
        r"""
from torch.nn.utils import parameters_to_vector as p2v

eta = 0.2
net = make_mlp(0)
for _ in range(1000):
    gd_step(net, eta)
lam, v = sharpness(net, None, iters=100)
u = torch.cat([t.flatten() for t in v]); u = u / u.norm()

thetas = []
for _ in range(40):
    thetas.append(p2v(net.parameters()).detach().clone())
    gd_step(net, eta)
T = torch.stack(thetas)
proj = (T - T.mean(0)) @ u
steps = T[1:] - T[:-1]
frac = ((steps @ u) ** 2 / (steps ** 2).sum(1)).mean().item()
print(f"S = {lam:.3f}, 2/eta = {2/eta}")
print("projections onto u:", np.round(proj[:10].numpy(), 4))
print(f"fraction of |step|^2 along u: {frac:.2f}")
"""
    ),
    answer(
        r"""
The projections alternate in sign with nearly constant magnitude, about $\pm0.04$ in this run: a period-2 oscillation along $u$, with $S$ within about 1% of $2/\eta$. About half of each step's squared length lies along the single direction $u$, out of about 4,800 parameters; the rest is the slow drift that actually reduces the loss. That is the picture the §3.2 toy encodes: one oscillating coordinate $x$ along $u$, plus slow coordinates.
"""
    ),
    # ================================================================== 3.2
    markdown(
        r"""
<a id="td23-2"></a>

## 3.2 Self-stabilization

Once $\eta S>2$, why doesn't the iterate blow up along the top eigenvector $u$? Because the oscillation lowers the sharpness: through the third derivative of the loss, an oscillation of size $x$ along $u$ also moves the iterate by $-\tfrac\eta2x^2\nabla S$ per step, down the gradient of the sharpness $S$.

**Paper:** Damian, Nichani & Lee, *Self-stabilization* (see [Readings](#td23-readings)).

The push is even in $x$, so it survives averaging over the period-2 bounce. When $\eta S>2$ the oscillation grows, the push grows with it, and $S$ is driven back to $2/\eta$. It all fits in a two-parameter toy, $L(x,y)=\tfrac12S(y)x^2+\ell(y)$ with $S'>0>\ell'$: $x$ oscillates along $u$, and $y$ is a slow coordinate along which the loss falls as the sharpness rises.

"""
        + FIG_TOY
        + r"""

<a id="td23-2-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Parameters $\theta$, Hessian $H$, sharpness $S(\theta)=\lambda_{\max}(H)$ with unit top eigenvector $u$, $\nabla S$ its gradient with respect to $\theta$. $\nabla^3L[a,b]$ is the third-derivative tensor contracted with two vectors (a vector). Write the iterate as a centre plus an oscillation along $u$: $\theta=\bar\theta+xu$.

**The expansion.** To second order, $\nabla L(\bar\theta+\delta)=\nabla L+H\delta+\tfrac12\nabla^3L[\delta,\delta]+O(\delta^3)$; with $\delta=xu$ this gives $xSu$ and $\tfrac12x^2\nabla^3L[u,u]$. By first-order eigenvalue perturbation ($S=u^\top Hu$; the variation of the unit eigenvector drops out), $\partial_iS=u^\top(\partial_iH)u$, so $\nabla S=\nabla^3L[u,u]$ (Damian et al., Lemma 2; needs $S$ simple), and

$$
\nabla L(\bar\theta+xu)=\nabla L(\bar\theta)+x\,S\,u+\tfrac12x^2\,\nabla S(\bar\theta)+O(x^3).
$$

The last term is even in $x$, so each step moves $\bar\theta$ by $-\tfrac\eta2x^2\nabla S$.

**The four-stage cycle** (Damian et al. §4). With $S\approx2/\eta+s$: (1) *progressive sharpening* raises $s$ by about $\eta\alpha$ per step, where $\alpha=-\langle\nabla L,\nabla S\rangle>0$, while $x$ is small; (2) *blowup*: once $s>0$, $|x|$ grows geometrically with alternating sign; (3) *self-stabilization*: the $\tfrac12x^2\nabla S$ term drives $s$ below $0$; (4) *return to stability*: $|x|$ shrinks, and the cycle repeats.

Their theorem (one unstable eigenvalue, progressive sharpening, bounded higher derivatives): gradient descent tracks *projected* gradient descent onto $\{S\le2/\eta,\ \nabla L\cdot u=0\}$, with loss $L(\theta^\dagger)+x^2/\eta$ for $\theta^\dagger$ the projected iterate. It implicitly solves $\min L$ subject to $S\le2/\eta$.

**The toy, concretely.** Damian et al. (Appendix B) use linear $S$ and $\ell$ plus a third coordinate. For the code we take $S(y)=y$ and $\ell(y)=c\,y^{-5}$, $c=62.5$: the loss keeps falling, with diminishing returns, as the sharpness grows (Exercise 3.2′ says why not linear $\ell$). Near $x=0$ the Hessian is $\operatorname{diag}(y,\ell''(y))$, so $y$ is the top eigenvalue while $y>30c\,y^{-7}$, i.e. $y\gtrsim2.6$; all runs start at $y_0\ge3$.

**Exercise 3.2 — Self-stabilization in the toy.** Use $L(x,y)=\tfrac12S(y)x^2+\ell(y)$ with $S'>0$ and $\ell'<0$.

(a) Write the gradient-descent maps for $(x,y)$. Show that when $\eta S(y)>2$, $|x|$ grows with alternating sign, and that the $y$-update gains a term that lowers $S$ and depends on $x$ only through $x^2$.

(b) Find the period-2 orbits $(x^*,y^*)\to(-x^*,y^*)\to(x^*,y^*)$. Show $S(y^*)=2/\eta$ and

$$
x^{*2}=-\frac{2\,\ell'(y^*)}{S'(y^*)} .
$$

(c) Check that (a) is the general mechanism: in the toy at $x=0$, what are the top eigenvector $u$, the sharpness gradient $\nabla S$ and the term $\tfrac12x^2\nabla S$ from the expansion above?

<details><summary>Hint</summary>

(a) $\partial_xL=S(y)x$ and $\partial_yL=\tfrac12S'(y)x^2+\ell'(y)$.

(b) $x\mapsto-x$ needs $1-\eta S(y^*)=-1$; $y$ fixed needs $\partial_yL=0$ at $(\pm x^*,y^*)$.
</details>
"""
    ),
    answer(
        r"""
**(a)** One step is

$$
x_{k+1}=\big(1-\eta S(y_k)\big)x_k,\qquad y_{k+1}=y_k-\eta\Big(\ell'(y_k)+\tfrac12S'(y_k)\,x_k^2\Big).
$$

If $\eta S>2$, the multiplier $1-\eta S<-1$, so $x$ changes sign each step and $|x|$ grows by the factor $\eta S-1>1$. The $y$-update has the sharpening term $-\eta\ell'>0$ plus $-\tfrac\eta2S'(y)x^2<0$, which lowers $y$ and hence $S$, is the same for $\pm x$, and eventually wins as $|x|$ grows.

**(b)** The map sends $x^*\mapsto(1-\eta S(y^*))x^*$, which equals $-x^*$ (with $x^*\neq0$) iff $S(y^*)=2/\eta$. The $y$-update vanishes at $(\pm x^*,y^*)$ iff $\ell'(y^*)+\tfrac12S'(y^*)x^{*2}=0$, i.e. $x^{*2}=-2\ell'(y^*)/S'(y^*)$, which is positive because $\ell'<0<S'$. Since $y$ does not move and $S(y^*)=2/\eta$, the next step sends $-x^*$ back to $x^*$: a period-2 orbit. It exists only if the sharpening pressure $-\ell'$ is positive where $S=2/\eta$; otherwise gradient descent simply converges below the edge.

**(c)** At $x=0$ the Hessian is $\operatorname{diag}\big(S(y),\ell''(y)\big)$, so (when $S>\ell''$) $u=e_x$ and the sharpness is $S(y)$, with $\nabla S=(0,S'(y))$. The term $\tfrac12x^2\nabla S=(0,\tfrac12S'x^2)$ is exactly the extra $y$-gradient in (a), and the linear term $xSu=(Sx,0)$ is the $x$-gradient. The toy is the general expansion with everything else set to zero.
"""
    ),
    markdown(
        r"""
**Code: the toy.** While $\eta S<2$, $x$ decays geometrically and in exact arithmetic gets absurdly small (from $x_0=10^{-3}$, $y_0=3$: about $10^{-139}$ by the edge at $\eta=0.3$), so the first blowup overshoots enormously and can kick $y$ through zero. In a real network the component along $u$ is constantly re-seeded (other directions, rotation of $u$, noise, round-off), so we add Gaussian noise of standard deviation $10^{-3}$ to $x$ each step (`noise=0` turns it off). The lower panel shows $|x_k|$ at $\eta=0.4$: at the noise floor while $\eta S<2$, a blowup once $\eta S$ crosses 2, an overshoot of $x^*$, then oscillation around $x^*$ in ever smaller cycles.
"""
    ),
    code(
        r"""
C_TOY = 62.5                      # ell(y) = C_TOY * y**-5, so S(y) = y and ell'(y) = -5 C_TOY y**-6

def toy_loss(x, y, c=C_TOY):
    return 0.5 * y * x * x + c / y**5

def toy_run(etas, y0=3.0, x0=1e-3, noise=1e-3, seed=0, c=C_TOY):
    # Gradient descent on L = y x^2/2 + c y^-5 with step etas[k] at step k, plus noise on x.
    # Returns arrays (x_k, y_k) of the state *before* step k; NaN after a divergence.
    rng = np.random.default_rng(seed)
    xi = noise * rng.standard_normal(len(etas))
    xs, ys = np.full(len(etas), np.nan), np.full(len(etas), np.nan)
    x, y = x0, y0
    with np.errstate(over="ignore", invalid="ignore"):
        for k, eta in enumerate(etas):
            xs[k], ys[k] = x, y
            x, y = (1 - eta * y) * x + xi[k], y - eta * (0.5 * x * x - 5 * c / y**6)
            if not (np.isfinite(x) and np.isfinite(y)) or y <= 0 or abs(x) > 1e3:
                break
    return xs, ys

# ---- demo
K = 4000
fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
for eta, col in zip((0.3, 0.4, 0.5), ("C0", "C1", "C2")):
    xs, ys = toy_run(np.full(K, eta))
    t = np.arange(K) * eta
    ax[0].plot(t, ys, color=col, lw=1, label=f"$\\eta$ = {eta}")
    ax[0].axhline(2 / eta, color=col, ls="--", lw=0.8)
    if eta == 0.4:
        ax[1].semilogy(t, np.abs(xs), color=col, lw=0.6, label="$|x_k|$, $\\eta$ = 0.4")
        ax[1].axhline(math.sqrt(10 * C_TOY * (eta / 2) ** 6), color="k", ls=":", lw=1, label="$x^*$")
    sigma2 = 10 * C_TOY * (eta / 2) ** 6      # x*^2 = -2 ell'(2/eta) / S'
    print(f"eta = {eta}: mean S (last 1000 steps) = {ys[-1000:].mean():.4f}, 2/eta = {2/eta:.4f};  "
          f"mean x^2 = {np.mean(xs[-1000:]**2):.5f}, x*^2 = {sigma2:.5f}")
ax[0].set(ylabel="$S = y$", title="toy: sharpness rises, overshoots, then settles or hovers at $2/\\eta$ (dashed)")
ax[1].set(xlabel="$t = k\\eta$", ylim=(1e-5, 3), xlim=(0, 600)); ax[0].legend(); ax[1].legend()
plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
**Exercise 3.2′ (optional) — When does the orbit attract?** These parts explain what "hovers at $2/\eta$" means. Take the toy with $S(y)=y$. Its period-2 orbit, from Exercise 3.2(b), is $(\pm x^*,y^*)$ with $y^*=2/\eta$ and $x^{*2}=2|\ell'(y^*)|$. Write $P=|\ell'(y^*)|$, $C=\ell''(y^*)$, $D=\eta C$, $Q=\eta^2x^{*2}=2\eta^2P$.

(d) Linearize the two-step map at the period-2 orbit. Show that the orbit is linearly (exponentially) attracting iff $|1-D+Q|<1$, $\;2-D+\tfrac Q2>0$ and $(D-\tfrac Q2)^2+\tfrac{Q^2}4-2D+2>0$, which for small $D,Q$ reduces to

$$
\ell''(y^*)>2\eta\,|\ell'(y^*)| .
$$

(e) In Damian et al.'s linear model ($\ell''=0$, so $D=0$), what are the eigenvalues of the two-step Jacobian? Show that the long-run averages are still pinned: if $y_k$ stays bounded then $\frac1K\sum_kx_k^2\to x^{*2}$ exactly, and if $|x_k|$ stays bounded away from $0$ and $\infty$ then $\frac1K\sum_k\log|1-\eta S(y_k)|\to0$. What does the second statement say about the average of $\eta S$?

(f) Check the condition of (d) exactly for $\ell=c\,y^{-5}$.

<details><summary>Hint</summary>

(d) With $S=y$, the one-step Jacobian at $(\pm x^*,y^*)$ is $J_\pm=\begin{pmatrix}-1&\mp\eta x^*\\\mp\eta x^*&1-D\end{pmatrix}$. Multiply $J_-J_+$, then use the Jury conditions for a real $2\times2$ matrix: both eigenvalues lie inside the unit circle iff $|\det J|<1$ and $|\operatorname{tr}J|<1+\det J$.

(e) Sum the $y$-updates over $K$ steps (they telescope), and take logs of the $x$-updates. For the average of $\eta S$, write $\eta S=2+\varepsilon$ and expand $\log|1-\eta S|$ in $\varepsilon$.
</details>
"""
    ),
    answer(
        r"""
**(d)** With $S=y$ and $x^{*2}=2P$,

$$
J_-J_+=\begin{pmatrix}-1&\eta x^*\\\eta x^*&1-D\end{pmatrix}\begin{pmatrix}-1&-\eta x^*\\-\eta x^*&1-D\end{pmatrix}
=\begin{pmatrix}1-Q&\eta x^*(2-D)\\-\eta x^*(2-D)&(1-D)^2-Q\end{pmatrix}.
$$

So $\operatorname{tr}J=2-2D+D^2-2Q$ and $\det J=(1-Q)\big((1-D)^2-Q\big)+Q(2-D)^2=(1-D+Q)^2$. The Jury conditions become

$$
|\det J|<1\iff|1-D+Q|<1,\qquad 1-\operatorname{tr}J+\det J=2Q\big(2-D+\tfrac Q2\big)>0,
$$
$$
1+\operatorname{tr}J+\det J=2\Big(\big(D-\tfrac Q2\big)^2+\tfrac{Q^2}4-2D+2\Big)>0 .
$$

For small $D,Q$ the last two hold automatically and the first is $Q<D$, i.e. $\eta^2x^{*2}<\eta\ell''$, i.e. $\ell''(y^*)>2\eta|\ell'(y^*)|$. Curvature of $\ell$ along $y$ damps deviations of $y$ (the $-D$); the discrete coupling between $x$ and $y$ anti-damps them (the $+Q$).

**(e)** With $D=0$, $J=\begin{pmatrix}1-Q&2\eta x^*\\-2\eta x^*&1-Q\end{pmatrix}$, whose eigenvalues are $1-Q\pm2i\sqrt Q$, of modulus $1+Q>1$: the orbit is a weakly unstable spiral, and the iterates circle it. (In the continuum limit the circles close: Damian et al.'s eq. 4 is a Lotka–Volterra-type system with a conserved quantity.)

The averages are pinned anyway. With $\ell'=-P$ constant and $S'=1$, summing the $y$-updates telescopes:

$$
y_K-y_0=-\eta\sum_{k<K}\big(\tfrac12x_k^2-P\big)\quad\Longrightarrow\quad\frac1K\sum_{k<K}x_k^2=2P-\frac{2(y_K-y_0)}{\eta K}\ \to\ 2P=x^{*2}
$$

if $y$ stays bounded. Taking logs of $|x_{k+1}|=|1-\eta y_k||x_k|$ gives $\frac1K\sum_k\log|1-\eta S_k|=\frac1K\log(|x_K|/|x_0|)\to0$. With $\eta S_k=2+\varepsilon_k$, $\log|1-\eta S_k|=\log(1+\varepsilon_k)=\varepsilon_k-\tfrac12\varepsilon_k^2+O(\varepsilon_k^3)$, so

$$
\langle\varepsilon\rangle=\tfrac12\langle\varepsilon^2\rangle+O(\langle|\varepsilon|^3\rangle)\ \ge0 :
$$

the time average of $\eta S$ is $2$ up to a second-order correction that puts it slightly *above* $2$, the more so the larger the oscillation of $S$. This is the sense in which the sharpness "hovers at, or just above, $2/\eta$" even when it never settles. The first identity is exact only because $\ell'$ and $S'$ are constant here; for nonlinear $\ell$ the same telescoping gives $\langle x^2\rangle=\langle-2\ell'(y)\rangle$, which equals $x^{*2}$ only approximately.

**(f)** For $\ell=c\,y^{-5}$: $P=5c\,y^{*-6}$ and $C=30c\,y^{*-7}$, so with $y^*=2/\eta$, $D=\eta C=15c\,\eta^2y^{*-6}=\tfrac32Q$ and $Q=10c\,\eta^2y^{*-6}=\tfrac{5c}{32}\eta^8$. Then $1-D+Q=1-\tfrac Q2$, so the first Jury condition is $0<Q<4$; the second is $2Q(2-Q)>0$, i.e. $Q<2$; the third is $\tfrac54Q^2-3Q+2>0$, which always holds (negative discriminant). So the orbit is linearly attracting iff $Q<2$, i.e. $\eta^8<64/(5c)$ (at $Q=2$ exactly a nonlinear analysis shows it still attracts, but only algebraically slowly): with $c=62.5$, $\eta<(0.2048)^{1/8}\approx0.82$. The figure's slider stops at $0.55$. (In invariant terms the small-$D,Q$ condition reads $\tilde\ell''(S)>\tfrac4S|\tilde\ell'(S)|$ at $S=2/\eta$, where $\tilde\ell$ is the loss as a function of the sharpness, so $\tilde\ell\propto S^{-p}$ satisfies it iff $p>3$, at every small enough $\eta$, while a quadratic $\ell$ satisfies it only within a factor $1.25$ of its minimum.) The attraction is slow: per two steps the deviation shrinks by $1-Q/2\approx0.997$ at $\eta=0.4$.
"""
    ),
    # ================================================================== 3.3
    markdown(
        r"""
<a id="td23-3"></a>

## 3.3 The central flow

At the edge the iterate bounces every step, but its time average moves smoothly. Can we write an ODE for that average, and predict the loss we measure? Yes: average the gradient over the oscillation, and choose the oscillation's size so that the sharpness stays at $2/\eta$.

**Paper:** Cohen, Damian, Talwalkar, Kolter & Lee, *Understanding optimization in deep learning with central flows* (see [Readings](#td23-readings)). Start with their [companion site](https://centralflows.github.io/), Part I in particular.

Let $w(t)$ be the time-averaged iterate (with $t=k\eta$), $S(w)$ its sharpness, $\nabla S$ the sharpness gradient, and $\sigma^2$ the variance of the oscillation along the top eigenvector. While $S<2/\eta$ the central flow is gradient flow. Once $S$ reaches $2/\eta$,

$$
\frac{dw}{dt}=-\Big[\nabla L(w)+\tfrac12\sigma^2(t)\,\nabla S(w)\Big],\qquad
\sigma^2=\max\!\Big(0,\ \frac{-2\langle\nabla L,\nabla S\rangle}{\|\nabla S\|^2}\Big).
$$

On the edge, $w$ follows gradient flow projected orthogonally to $\nabla S$, sliding along the surface $S=2/\eta$, and the predicted time-averaged loss is $L(w)+\sigma^2/\eta$. It is a heuristic, tested on real networks over long runs (full batch, no momentum; better for MSE than cross-entropy; degrading as $\eta$ grows; possibly poor for ReLU; blind to large loss spikes).

<a id="td23-3-walkthrough"></a>

**Derivations and walkthrough.** *From §3.2:* With $u$ the unit top eigenvector at $w$, $\nabla L(w+xu)=\nabla L(w)+xSu+\tfrac12x^2\nabla S(w)+O(x^3)$. Cohen et al. count time in steps, so they write $dw/dt=-\eta[\nabla L+\tfrac12\sigma^2\nabla S]$.

**Several unstable directions.** Then $\sigma^2$ becomes the oscillation's covariance $\Sigma\succeq0$, the drift is $-\nabla L-\tfrac12\nabla\operatorname{tr}(H\Sigma)$ (gradient with $\Sigma$ fixed), and the loss is $L+\operatorname{tr}\Sigma/\eta$. $\Sigma$ lives only in directions whose eigenvalue sits at $2/\eta$ and keeps them from crossing it: a *complementarity problem*, the matrix version of "either $\sigma^2=0$ or the constraint is tight".

**Exercise 3.3 — Derive the one-direction central flow.** Let $w$ be the centre of the oscillation, and suppose the oscillation is $\delta=xu$ with $\mathbb E[x]=0$ and $\mathbb E[x^2]=\sigma^2$.

(a) Show that $\mathbb E\,\nabla L(w+\delta)=\nabla L(w)+\tfrac12\sigma^2\nabla S(w)+O(\mathbb E|x|^3)$. Gradient descent moves $w$ by $-\eta$ times this per step: write the flow in $t=k\eta$.

(b) Suppose $S(w)=2/\eta$. Find the $\sigma^2$ for which $\frac{d}{dt}S(w(t))=0$. Show $\sigma^2\ge0$ iff $\langle\nabla L,\nabla S\rangle\le0$, i.e. iff plain gradient flow would increase $S$. What should the flow do when this fails?

(c) Show that while the constraint is active the flow is $\dot w=-\Pi^\perp\nabla L$, with $\Pi^\perp=I-\nabla S\nabla S^\top/\|\nabla S\|^2$, and that $L(w(t))$ is non-increasing.

(d) Show that the time-averaged loss is $\mathbb E\,L(w+\delta)\approx L(w)+\sigma^2/\eta$.

(e) Evaluate $\sigma^2$ for the §3.2 toy $L(x,y)=\tfrac12S(y)x^2+\ell(y)$ at $(0,y^*)$, where $S(y^*)=2/\eta$, and compare with the orbit amplitude $x^{*2}=-2\ell'(y^*)/S'(y^*)$ from Exercise 3.2(b).

<details><summary>Hint</summary>

(b) $\frac{dS}{dt}=\langle\nabla S,\dot w\rangle$. (c) Substitute your $\sigma^2$; then $\frac{dL}{dt}=\langle\nabla L,\dot w\rangle$ and $\Pi^\perp$ is a symmetric idempotent matrix. (d) Expand $L$ to second order around $w$ and use $u^\top Hu=S=2/\eta$.
</details>
"""
    ),
    answer(
        r"""
**(a)** From $\nabla L(w+xu)=\nabla L(w)+xSu+\tfrac12x^2\nabla^3L[u,u]+O(x^3)$ and $\nabla^3L[u,u]=\nabla S$, taking expectations kills the term linear in $x$:

$$
\mathbb E\,\nabla L(w+xu)=\nabla L(w)+\tfrac12\sigma^2\nabla S(w)+O(\mathbb E|x|^3).
$$

Per step the centre moves by $-\eta$ times this, which over time $\eta$ is the flow $\dot w=-\big[\nabla L+\tfrac12\sigma^2\nabla S\big]$ with $t=k\eta$.

**(b)** $\frac{dS}{dt}=\langle\nabla S,\dot w\rangle=-\langle\nabla S,\nabla L\rangle-\tfrac12\sigma^2\|\nabla S\|^2$. Setting this to zero:

$$
\sigma^2=\frac{-2\langle\nabla L,\nabla S\rangle}{\|\nabla S\|^2}.
$$

This is non-negative iff $\langle\nabla L,\nabla S\rangle\le0$. Under gradient flow $\dot S=-\langle\nabla S,\nabla L\rangle$, so the condition says gradient flow would *raise* $S$: progressive sharpening is what keeps the oscillation alive. If instead $\langle\nabla L,\nabla S\rangle>0$, gradient flow already lowers $S$, no oscillation is needed, and the flow leaves the edge with $\sigma^2=0$. Combining both cases gives the $\max(0,\cdot)$.

**(c)** Substituting,

$$
\dot w=-\nabla L+\frac{\langle\nabla L,\nabla S\rangle}{\|\nabla S\|^2}\nabla S=-\Big(I-\frac{\nabla S\nabla S^\top}{\|\nabla S\|^2}\Big)\nabla L=-\Pi^\perp\nabla L .
$$

Then $\frac{dL}{dt}=-\nabla L^\top\Pi^\perp\nabla L=-\|\Pi^\perp\nabla L\|^2\le0$, using $\Pi^\perp=(\Pi^\perp)^\top=(\Pi^\perp)^2$. The flow slides downhill along the level set $S=2/\eta$. This is close to Damian et al.'s projected gradient descent, but not identical: their constraint set also requires $\nabla L\cdot u=0$, a condition Cohen et al. drop (their §3.2).

**(d)** $L(w+xu)=L(w)+x\nabla L^\top u+\tfrac12x^2u^\top Hu+O(x^3)$. Averaging, the linear term drops out and $u^\top Hu=S=2/\eta$:

$$
\mathbb E\,L(w+\delta)\approx L(w)+\tfrac12\sigma^2S(w)=L(w)+\frac{\sigma^2}{\eta}.
$$

The measured loss sits above the loss at the centre by $\sigma^2/\eta$ (compare Damian et al.'s $L(\theta^\dagger)+x^2/\eta$).

**(e)** In the toy at $(0,y)$: $\nabla L=(0,\ell'(y))$ and $\nabla S=(0,S'(y))$, so $\sigma^2=-2\ell'(y)/S'(y)$, and at $y^*$ this is exactly $x^{*2}$ from 3.2(b): the central flow's oscillation variance is the squared amplitude of the period-2 orbit. (When the iterates circle the orbit instead of settling, 3.2′(e) shows the long-run average of $x_k^2$ is still $x^{*2}$, exactly in the linear model and approximately otherwise.) The central flow of the 2-D toy is trivial (it stops at $y^*$); the code below adds a third coordinate so that it has somewhere to go.
"""
    ),
    markdown(
        r"""
**Code: central flow against gradient descent.** Does the central flow track gradient descent when the edge is curved? Add a coordinate $z$ to the toy: $L(x,y,z)=\tfrac12S(y,z)\,x^2+\ell(y,z)$ with

$$
S(y,z)=y\,(1+\kappa z),\qquad \ell(y,z)=c\,y^{-5}+\tfrac m2z^2,\qquad \kappa=0.3,\ m=0.002 ,
$$

and $c=62.5$. Now $\nabla S=(1+\kappa z,\ \kappa y)$ rotates, and the flow slides along the curved surface $y(1+\kappa z)=2/\eta$: lowering $z$ lets $y$ grow at fixed sharpness, lowering $c\,y^{-5}$. In gradient descent the $x^2$ kick does the sliding (its $z$-component is $-\tfrac\eta2\kappa y\,x^2$). We integrate the central flow (Euler, $dt=\eta/20$, projected back onto the surface each step) and compare $y$, $z$, oscillation variance and loss, averaged over matching windows because the iterates circle the central point slowly.
"""
    ),
    code(
        r"""
eta, kap, m, K = 0.4, 0.3, 0.002, 4000
S3  = lambda y, z: y * (1 + kap * z)
gS3 = lambda y, z: np.array([1 + kap * z, kap * y])            # grad S in (y, z)
l3  = lambda y, z: C_TOY / y**5 + 0.5 * m * z * z
gl3 = lambda y, z: np.array([-5 * C_TOY / y**6, m * z])         # grad ell in (y, z)

# gradient descent on (x, y, z), same noise floor on x as before
rng = np.random.default_rng(0)
x, y, z = 1e-3, 3.0, 2.0
G = np.zeros((K, 5))                                          # columns: x, y, z, loss, S
for k in range(K):
    G[k] = x, y, z, 0.5 * S3(y, z) * x * x + l3(y, z), S3(y, z)
    gy, gz = gl3(y, z) + 0.5 * x * x * gS3(y, z)
    x, y, z = (1 - eta * S3(y, z)) * x + 1e-3 * rng.standard_normal(), y - eta * gy, z - eta * gz

# central flow in t = k*eta
dt, w, CF = eta / 20, np.array([3.0, 2.0]), []
for i in range(K * 20):
    gl, gS = gl3(*w), gS3(*w)
    on_edge = S3(*w) >= (2 / eta) * (1 - 1e-9)                # tolerance: the projection lands exactly on the edge
    s2 = max(0.0, -2 * gl @ gS / (gS @ gS)) if on_edge else 0.0
    if i % 20 == 0:
        CF.append((w[0], w[1], s2, l3(*w), l3(*w) + s2 / eta))
    w = w - dt * (gl + 0.5 * s2 * gS)
    if s2 > 0:
        w[0] = (2 / eta) / (1 + kap * w[1])                   # back onto S = 2/eta
CF = np.array(CF)

idx, x2m = running_mean(G[:, 0] ** 2, 200)
_, lossm = running_mean(G[:, 3], 200)
t = np.arange(K) * eta
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(t, G[:, 1], color="C0", lw=1.2, label="GD $y_k$"); ax[0].plot(t, CF[:, 0], "k--", lw=0.8, label="central flow $y$")
ax[0].plot(t, G[:, 2], color="C2", lw=0.8, label="GD $z_k$"); ax[0].plot(t, CF[:, 1], "k:", label="central flow $z$")
ax[0].set(xlabel="$t$", title="trajectory"); ax[0].legend(fontsize=7)
ax[1].plot(t[idx], x2m, color="C1", label="GD $x^2$, 200-step mean"); ax[1].plot(t, CF[:, 2], "k--", label="$\\sigma^2(t)$")
ax[1].set(xlabel="$t$", ylim=(0, 0.02), title="oscillation variance"); ax[1].legend(fontsize=7)
ax[2].plot(t[idx], lossm, color="C3", label="GD loss, 200-step mean"); ax[2].plot(t, CF[:, 4], "k--", label="$L(w)+\\sigma^2/\\eta$")
ax[2].plot(t, CF[:, 3], "k:", label="$L(w)$"); ax[2].set(xlabel="$t$", ylim=(0, 0.05), title="loss"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
# Early: 200-step windows. Late, the iterates circle the central point in slow, wide loops,
# so compare means over 1000-step windows (CF values averaged over the same window).
for k, half in ((400, 100), (800, 100), (2000, 500), (3000, 500)):
    w_ = slice(k - half, k + half)
    gd = [G[w_, i].mean() for i in (1, 2)] + [(G[w_, 0] ** 2).mean(), G[w_, 3].mean()]
    cf = [CF[w_, i].mean() for i in (0, 1, 2, 4)]
    print(f"t = {k*eta:5.0f} ({2*half}-step window): y {gd[0]:.3f} vs CF {cf[0]:.3f}; z {gd[1]:.3f} vs {cf[1]:.3f}; "
          f"x^2 {gd[2]:.5f} vs sigma^2 {cf[2]:.5f}; loss {gd[3]:.5f} vs L+sigma^2/eta {cf[3]:.5f}")
"""
    ),
    markdown(
        r"""
Late in the run, 1000-step averages of all four quantities agree with the central flow to a few percent; just after the first excursion, variance and loss are off by about 10%. The measured loss is more than twice $L(w)$: much of what you see during EoS is the oscillation. The toy satisfies the flow's assumptions by construction, so this checks the derivation, not the claim about networks.
"""
    ),
    # ================================================================== 3.4 Rod / principal
    markdown(
        r"""
<a id="td23-4"></a>

## 3.4 Rod Flow and principal flow

The central flow needs the top eigenvector, the sharpness gradient and, with several unstable directions, a constrained matrix problem. Is there a continuous-time model of the edge needing only gradients and Hessian-vector products? *Rod Flow* is one. And *principal flow* is what the small-step modified-flow series of §3.1 becomes if every order is kept.

**Papers:** Regis & Chewi, *Rod Flow* (2026); Rosca, Wu, Qin & Dherin, the principal flow (2023) (see [Readings](#td23-readings)).

Rod Flow tracks the *rod* joining two consecutive iterates (midpoint and half-length) instead of the iterate: at the edge the iterates jump back and forth, but the rod moves smoothly, and its length carries the instability. Principal flow applies the exact quadratic flow of Exercise 3.1, rate $\frac1\eta\log(1-\eta\lambda)$, eigendirection by eigendirection; it captures the onset of instability but, having no third derivatives, not the stabilization.

<a id="td23-4-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* Gradient-descent iterates $w_{k+1}=w_k-\eta\nabla L(w_k)$, continuous time $t=k\eta$, rod midpoint $\bar w_k=\tfrac12(w_{k+1}+w_k)$, half-length $\delta_k=\tfrac12(w_{k+1}-w_k)=-\tfrac\eta2\nabla L(w_k)$, and $\Sigma_k=\delta_k\delta_k^\top$ (the rod's squared length, as a matrix). The midpoint's update is *exact*,

$$
\bar w_{k+1}-\bar w_k=-\tfrac\eta2\big[\nabla L(\bar w_k+\delta_k)+\nabla L(\bar w_k-\delta_k)\big],
$$

with gradients at the rod's ends, where the iterates are, not Taylor-expanded around the centre. $\Sigma_k$ also obeys an exact identity (Exercise 3.4(a)). As ODEs, with one backward-error correction for the midpoint, in $t=k\eta$ and with $L_\pm=L(\bar w\pm\delta)$:

$$
\frac{d\bar w}{dt}=-\tfrac12\big[\nabla L_++\nabla L_-\big]-\tfrac\eta8\big[\nabla^2L_++\nabla^2L_-\big]\big[\nabla L_++\nabla L_-\big],\qquad
\frac{d\Sigma}{dt}=\tfrac\eta4\big[\nabla L_+\nabla L_+^\top+\nabla L_-\nabla L_-^\top\big]-\tfrac2\eta\Sigma .
$$

To close the system, $\delta$ is $\sqrt{\lambda_{\max}(\Sigma)}$ times $\Sigma$'s top eigenvector (the authors observe $\Sigma$ stays nearly rank one). On a quadratic this reproduces the $2/\eta$ threshold (Exercise 3.4), on a quartic it gives a finite stable rod length, and expanding $\nabla L_\pm$ to second order heuristically recovers the central flow.

**Exercise 3.4 — Rod Flow on a quadratic.**

(a) Verify the exact midpoint update above, and $\delta_k=-\tfrac\eta2\nabla L(w_k)$. Then show that $\Sigma_k=\delta_k\delta_k^\top$ satisfies, exactly,

$$
\Sigma_{k+1}-\Sigma_k=\tfrac{\eta^2}4\big[\nabla L_+\nabla L_+^\top+\nabla L_-\nabla L_-^\top\big]-2\Sigma_k,\qquad L_\pm=L(\bar w_k\pm\delta_k),
$$

and that dividing by the time step $\eta$ gives the $\Sigma$ equation above.

(b) Take $L=\tfrac12Sw^2$ in one dimension and $\Sigma=\delta^2$. Write the $\Sigma$ equation at $\bar w=0$ and show $\Sigma$ grows iff $\eta S>2$.

(c) Substitute $\delta=0$ into the $\bar w$ equation and show it becomes the modified flow $\dot{\bar w}=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$. Is $\delta=0$ a consistent state of the full system away from a stationary point?

<details><summary>Hint</summary>

(a) $w_k=\bar w_k-\delta_k$ and $w_{k+1}=\bar w_k+\delta_k$; then $\bar w_{k+1}-\bar w_k=\tfrac12(w_{k+2}-w_k)$. For $\Sigma$, write $\delta_{k+1}$ and $\delta_k$ in terms of $\nabla L_\pm$. (b) $\nabla L_\pm=S(\bar w\pm\delta)$. (c) Look at $d\Sigma/dt$ at $\Sigma=0$.
</details>
"""
    ),
    answer(
        r"""
**(a)** By definition $w_{k+1}-w_k=-\eta\nabla L(w_k)$, so $\delta_k=-\tfrac\eta2\nabla L(w_k)$. Also $\bar w_k\pm\delta_k$ are $w_{k+1}$ and $w_k$. Then

$$
\bar w_{k+1}-\bar w_k=\tfrac12(w_{k+2}-w_k)=\tfrac12\big[(w_{k+2}-w_{k+1})+(w_{k+1}-w_k)\big]=-\tfrac\eta2\big[\nabla L(w_{k+1})+\nabla L(w_k)\big],
$$

which is $-\tfrac\eta2[\nabla L(\bar w_k+\delta_k)+\nabla L(\bar w_k-\delta_k)]$. No approximation.

For the rod's length: $\delta_{k+1}=-\tfrac\eta2\nabla L(w_{k+1})=-\tfrac\eta2\nabla L_+$, so $\Sigma_{k+1}=\tfrac{\eta^2}4\nabla L_+\nabla L_+^\top$; and $\delta_k=-\tfrac\eta2\nabla L(w_k)=-\tfrac\eta2\nabla L_-$, so $\Sigma_k=\tfrac{\eta^2}4\nabla L_-\nabla L_-^\top$. Hence

$$
\Sigma_{k+1}-\Sigma_k=\tfrac{\eta^2}4\nabla L_+\nabla L_+^\top-\Sigma_k=\tfrac{\eta^2}4\big[\nabla L_+\nabla L_+^\top+\nabla L_-\nabla L_-^\top\big]-2\Sigma_k ,
$$

exactly; the symmetric form treats both ends alike, so it is unchanged when the oscillation flips sign. Reading the left side as $\eta\,d\Sigma/dt$ gives $d\Sigma/dt=\tfrac\eta4[\dots]-\tfrac2\eta\Sigma$. (The decay rate $2/\eta$ is fast compared with the drift of $\bar w$, which is why the authors skip a backward-error correction for $\Sigma$.)

**(b)** With $\nabla L_\pm=S(\bar w\pm\delta)$ and $\bar w=0$: $\nabla L_+^2+\nabla L_-^2=2S^2\delta^2=2S^2\Sigma$. So

$$
\frac{d\Sigma}{dt}=\frac\eta4\cdot2S^2\Sigma-\frac2\eta\Sigma=\Big(\frac{\eta S^2}2-\frac2\eta\Big)\Sigma ,
$$

which grows iff $\eta S^2/2>2/\eta$, i.e. $\eta S>2$. (In the paper's step-counting time this is $d\Sigma/d\tau=(\eta^2S^2/2-2)\Sigma$, their §5.2.) The rod's length, not its midpoint, carries the instability: Rod Flow escapes the obstruction of Exercise 3.1 because it has a second real variable.

**(c)** At $\delta=0$, $\nabla L_\pm=\nabla L$ and $\nabla^2L_\pm=H$, so $\dot{\bar w}=-\tfrac12\cdot2\nabla L-\tfrac\eta8\cdot2H\cdot2\nabla L=-\nabla L-\tfrac\eta2H\nabla L=-\nabla\big(L+\tfrac\eta4\|\nabla L\|^2\big)$: the implicit-gradient-regularization flow. But $\delta=0$ is not consistent away from a stationary point: at $\Sigma=0$, $d\Sigma/dt=\tfrac\eta2\nabla L\nabla L^\top\neq0$. Away from the edge the rod relaxes to a small length set by the step itself ($\delta=-\tfrac\eta2\nabla L$; on a linear loss $L=-b^\top w$ the equilibrium is $\Sigma_*=\tfrac{\eta^2}4bb^\top$, the paper's §5.1 example). So the IGR flow is the leading behaviour when the rod is short compared with the scale on which $\nabla L$ varies, not an exact sub-case.
"""
    ),
    # ================================================================== 3.5 adaptive
    markdown(
        r"""
<a id="td23-5"></a>

## 3.5 Adaptive optimizers

Does Adam have an edge of stability? Yes, but in the *preconditioned* sharpness $\lambda_{\max}(P^{-1}H)$, where $H$ is the loss Hessian and $P$ is Adam's diagonal preconditioner. In full-batch training with $\beta_1=0.9$ it sits at $38/\eta$ while the raw sharpness keeps rising.

**Paper:** Cohen, Ghorbani, Krishnan, Agarwal et al., *Adaptive gradient methods at the edge of stability* (v2; see [Readings](#td23-readings)).

With $P$ frozen, Adam on a quadratic is heavy-ball momentum on a rescaled problem, stable iff $\lambda_{\max}(P^{-1}H)<2(1+\beta_1)/\big((1-\beta_1)\eta\big)=38/\eta$. The edge holds because the instability's large gradients inflate $P$, pulling the preconditioned sharpness back down. With minibatches it sits below the threshold, more so for small batches (and at very small $\eta$, even in full batch).

<a id="td23-5-walkthrough"></a>

**Derivations and walkthrough.** *Prerequisites:* Heavy-ball momentum is $x_{k+1}=x_k-\eta\nabla L(x_k)+\beta(x_k-x_{k-1})$. On $\tfrac12\lambda x^2$ its characteristic polynomial is $z^2-(1+\beta-\eta\lambda)z+\beta$, and for $0\le\beta<1$ both roots lie inside the unit circle iff $0<\eta\lambda<2(1+\beta)$ (derivation and pictures: [Goh, *Why momentum really works*](https://distill.pub/2017/momentum/)). Adam with gradient $g_k$: first moment $m_k=\beta_1m_{k-1}+(1-\beta_1)g_k$ (an exponential moving average, EMA), second moment $v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2$ (coordinatewise), step $\theta_{k+1}=\theta_k-\eta P_k^{-1}m_k$ with $P_k=\operatorname{diag}(\sqrt{v_k})+\epsilon I$.

**Exercise 3.5 — The $38/\eta$ threshold.**

(a) Show that EMA momentum, $m_k=\beta_1m_{k-1}+(1-\beta_1)\nabla L(\theta_k)$, $\theta_{k+1}=\theta_k-\eta m_k$, is heavy-ball momentum with step $\eta(1-\beta_1)$ and momentum $\beta_1$. Deduce that on $\tfrac12\lambda\theta^2$ with $\lambda>0$ it is stable iff $\lambda<2(1+\beta_1)/\big((1-\beta_1)\eta\big)$, and evaluate this at $\beta_1=0.9$.

(b) Now use a fixed positive diagonal preconditioner, $\theta_{k+1}=\theta_k-\eta P^{-1}m_k$, on $\tfrac12\theta^\top H\theta$ with $H$ positive definite. Show that stability is governed by the eigenvalues of $P^{-1}H$, which are real and positive even though $P^{-1}H$ is not symmetric, and that the condition is $\lambda_{\max}(P^{-1}H)<38/\eta$.

(c) Explain why, if Adam's raw sharpness keeps rising, its preconditioned sharpness can stay at $38/\eta$ instead of overshooting and diverging. Be careful about *which* direction is unstable.

<details><summary>Hint</summary>

(a) Write $m_{k-1}=(\theta_{k-1}-\theta_k)/\eta$ and substitute. (b) Change variables to $z=P^{1/2}\theta$ and $\mu=P^{-1/2}m$. (c) The unstable mode is an eigenvector $q$ of $P^{-1/2}HP^{-1/2}$, i.e. $\delta\theta\propto P^{-1/2}q$. What gradient does it produce, and where does that gradient land in $v$?
</details>
"""
    ),
    answer(
        r"""
**(a)** From $\theta_k=\theta_{k-1}-\eta m_{k-1}$, $m_{k-1}=(\theta_{k-1}-\theta_k)/\eta$. Then

$$
\theta_{k+1}=\theta_k-\eta\big[\beta_1m_{k-1}+(1-\beta_1)\nabla L(\theta_k)\big]=\theta_k-\eta(1-\beta_1)\nabla L(\theta_k)+\beta_1(\theta_k-\theta_{k-1}),
$$

which is heavy ball with step $\eta(1-\beta_1)$ and momentum $\beta_1$. The heavy-ball condition $0<\eta(1-\beta_1)\lambda<2(1+\beta_1)$ gives

$$
\lambda<\frac{2(1+\beta_1)}{(1-\beta_1)\eta}=\frac{2\cdot1.9}{0.1\,\eta}=\frac{38}{\eta}\quad(\beta_1=0.9).
$$

The extra room compared with plain heavy ball comes entirely from the factor $1-\beta_1$ on the gradient.

**(b)** Set $z=P^{1/2}\theta$ and $\mu=P^{-1/2}m$. Then $z_{k+1}=z_k-\eta P^{1/2}P^{-1}m_k=z_k-\eta\mu_k$, and $\mu_k=\beta_1\mu_{k-1}+(1-\beta_1)P^{-1/2}H\theta_k=\beta_1\mu_{k-1}+(1-\beta_1)\tilde Hz_k$ with $\tilde H=P^{-1/2}HP^{-1/2}$. So in $z$ coordinates this is EMA momentum on the quadratic $\tfrac12z^\top\tilde Hz$, which decouples along the eigenvectors of the symmetric positive-definite matrix $\tilde H$; each mode is stable iff its eigenvalue lies in $(0,38/\eta)$. Finally $\tilde H=P^{1/2}(P^{-1}H)P^{-1/2}$ is similar to $P^{-1}H$, so they share eigenvalues, which are real and positive because $\tilde H$ is symmetric positive definite. Hence the condition is $\lambda_{\max}(P^{-1}H)<38/\eta$. (Cohen et al. prove the same equivalence in their App. A.2. Without positive definiteness one also needs every eigenvalue positive.)

**(c)** $P$ is not frozen: $v_k$ is a moving average of squared gradients. Let $q$ be the top eigenvector of $\tilde H$, with eigenvalue $\lambda_{\rm eff}=\lambda_{\max}(P^{-1}H)$. The unstable displacement is $\delta\theta=x\,P^{-1/2}q$, which in general is *not* the raw top eigenvector $u$ (for $H=\operatorname{diag}(2,1)$, $P=\operatorname{diag}(100,1)$ the raw and preconditioned sharpest directions are different coordinates). It changes the gradient by $H\delta\theta=x\lambda_{\rm eff}\,P^{1/2}q$. When $\lambda_{\rm eff}$ crosses $38/\eta$, $|x|$ grows geometrically and so does this gradient; its coordinatewise square inflates $v$, and hence $P$, in exactly the coordinates where $P^{1/2}q$ has weight. Increasing $P$ there lowers $\lambda_{\max}(P^{-1}H)$; the growth stops once it is back below $38/\eta$, the large gradients die out, $v$ decays at rate $1-\beta_2$, and progressive sharpening pushes the preconditioned sharpness back up. This negative feedback equilibrates at the threshold. Raw sharpness can keep growing because each increase in $H$ along the unstable mode is matched by an increase in $P$. So Adam has two stabilizing channels: the third-derivative push of §3.2, and the more direct response of $P$ to large gradients.
"""
    ),
    code(
        r"""
# Check the 38/eta threshold for EMA momentum (beta1 = 0.9), plain and with a frozen diagonal preconditioner.
def ema_final(eta_H, P, beta=0.9, steps=4000, seed=0):
    th = np.random.default_rng(seed).standard_normal(len(P)); m = np.zeros_like(th)
    for _ in range(steps):
        m = beta * m + (1 - beta) * (eta_H @ th)        # eta folded into H: eta_H = eta * H
        th = th - m / P
        if np.linalg.norm(th) > 1e100:
            return np.inf                               # diverged
    return np.linalg.norm(th)

for r in (37.5, 38.5):
    print(f"1-D, eta*lambda = {r}: |theta| after 4000 steps = {ema_final(np.array([[r]]), np.ones(1)):.2e}")
A = np.random.default_rng(1).standard_normal((3, 3)); H = A @ A.T + 0.1 * np.eye(3)   # positive definite
P = np.array([0.5, 2.0, 5.0])
lam_pre = np.max(np.linalg.eigvals(np.diag(1 / P) @ H).real)
for r in (0.98, 1.02):
    eta = r * 38 / lam_pre
    print(f"3-D, eta * lambda_max(P^-1 H) = {r*38:.2f} (raw eta*S = {eta*np.linalg.eigvalsh(H).max():.1f}): "
          f"|theta| = {ema_final(eta * H, P):.2e}")
"""
    ),
    markdown(
        r"""
**Code: Adam's edge in the MLP.** Full-batch Adam on the §3.1 network ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-7}$, no bias correction, as in Cohen et al.) at two step sizes. Every 100 steps we freeze $P$ and compute $\lambda_{\max}(P^{-1/2}HP^{-1/2})$ and the raw $S$ by Lanczos on Hessian-vector products.
"""
    ),
    code(
        r"""
import scipy.sparse.linalg as sla

def top_eig_lanczos(net, Pis=None):
    # Largest eigenvalue of P^{-1/2} H P^{-1/2} (or of H if Pis is None) by Lanczos on Hessian-vector products.
    # Power iteration converges too slowly here (small spectral gap), so we use scipy's eigsh.
    params = list(net.parameters()); shapes = [q.shape for q in params]; sizes = [q.numel() for q in params]
    grads = torch.autograd.grad(mse(net), params, create_graph=True)
    pis = None if Pis is None else torch.cat([q.flatten() for q in Pis]).double().numpy()
    def matvec(x):
        x = np.asarray(x).ravel() * (1 if pis is None else pis)
        vs = [torch.tensor(c, dtype=torch.float32).view(sh) for c, sh in zip(np.split(x, np.cumsum(sizes)[:-1]), shapes)]
        Hv = torch.autograd.grad(grads, params, grad_outputs=vs, retain_graph=True)
        y = torch.cat([h.flatten() for h in Hv]).double().numpy()
        return y * (1 if pis is None else pis)
    N = sum(sizes)
    op = sla.LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    return sla.eigsh(op, k=1, which="LA", tol=1e-5, return_eigenvectors=False)[0]

def run_adam(eta, steps=1500, b1=0.9, b2=0.999, eps=1e-7, every=100):
    net = make_mlp(0); params = list(net.parameters())
    m = [torch.zeros_like(q) for q in params]; vv = [torch.zeros_like(q) for q in params]
    rec = []
    for k in range(steps + 1):
        g = torch.autograd.grad(mse(net), params)
        with torch.no_grad():
            for i in range(len(params)):
                m[i].mul_(b1).add_((1 - b1) * g[i]); vv[i].mul_(b2).add_((1 - b2) * g[i] ** 2)
        if k > 0 and k % every == 0:
            Pis = [1 / (vi.sqrt() + eps).sqrt() for vi in vv]       # frozen P = diag(sqrt v) + eps
            rec.append((k, top_eig_lanczos(net), top_eig_lanczos(net, Pis)))
        with torch.no_grad():
            for i, q in enumerate(params):
                q -= eta * m[i] / (vv[i].sqrt() + eps)
    return np.array(rec)

t0 = time.time()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for eta, col in ((1e-3, "C0"), (3e-3, "C1")):
    r = run_adam(eta)
    ax[0].plot(r[:, 0], eta * r[:, 2], color=col, label=f"$\\eta$ = {eta:g}")
    ax[1].plot(r[:, 0], r[:, 1], color=col, label=f"$\\eta$ = {eta:g}")
    late = r[r[:, 0] > 500]
    print(f"eta = {eta:g}, steps 600-1500: eta * lambda_max(P^-1 H) mean {eta * late[:, 2].mean():.1f}, "
          f"range [{eta * late[:, 2].min():.1f}, {eta * late[:, 2].max():.1f}] (threshold 38);  "
          f"raw S = {late[:, 1].mean():.1f}, so eta * S = {eta * late[:, 1].mean():.3f}")
ax[0].axhline(38, color="k", ls="--", lw=1)
ax[0].set(xlabel="step", ylabel="$\\eta\\,\\lambda_{\\max}(P^{-1}H)$", title="preconditioned sharpness (dashed: 38)")
ax[1].set(xlabel="step", ylabel="raw sharpness $S$", title="raw sharpness: far below $2/\\eta$")
ax[0].legend(); plt.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f} s)")
"""
    ),
    markdown(
        r"""
At both step sizes $\eta\lambda_{\max}(P^{-1}H)$ hovers near 38 (about 36 to 38.6), while $\eta S$ is a few hundredths: the edge is in the preconditioned quantity. A caveat on *how* it gets there: here the loss is near zero after a few hundred steps, so $v$ (hence $P$) decays, raising $\lambda_{\max}(P^{-1}H)$ toward the edge, and each renewed oscillation's gradient burst pushes it back. That is not progressive sharpening; in Cohen et al.'s longer runs Adam is still training at the edge while the raw sharpness climbs.
"""
    ),
    # ================================================================== 3.6 EoSS
    markdown(
        r"""
<a id="td23-6"></a>

## 3.6 The edge of stochastic stability

Under minibatch SGD, does the sharpness still rise to $2/\eta$? The full-batch $\lambda_{\max}$ settles lower, by a gap that grows as the batch size $b$ shrinks. What sits at $2/\eta$, for every batch size, is the **batch sharpness**

$$
\mathrm{BS}(\theta)=\mathbb E_B\!\left[\frac{g_B^\top H_B\,g_B}{\|g_B\|^2}\right],
$$

where a minibatch $B$ has gradient $g_B$ and Hessian $H_B$: each minibatch's curvature along its own gradient, with the expectation *outside* the ratio.

**Paper:** Andreyev & Beneventano, *Edge of Stochastic Stability* (v5; see [Readings](#td23-readings)).

The subtlety is which average: requiring the *expected* batch-loss change to be negative gives a ratio of expectations instead, which also stabilizes, but above $2/\eta$ and at a batch-size-dependent level. Why batch sharpness settles *at* $2/\eta$ (an SGD analogue of §3.2) is open.

<a id="td23-6-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* A minibatch $B$ of size $b$ has loss $L_B$, gradient $g_B=\nabla L_B$ and Hessian $H_B$; the SGD step is $\theta'=\theta-\eta g_B$; $S_B=g_B^\top H_Bg_B/\|g_B\|^2$, so $\mathrm{BS}=\mathbb E[S_B]$. The paper's theory is one-sided; Exercise 3.6(c–d) examines what it shows.

**Exercise 3.6 — Which average?** Fix $\theta$ and draw a minibatch $B$.

(a) Expand $L_B(\theta')$ to second order. Show that the step decreases its own batch loss iff $S_B<2/\eta$.

(b) Show that $\mathbb E_B\big[L_B(\theta')-L_B(\theta)\big]<0$ (to the same order) iff $\mathbb E[g_B^\top H_Bg_B]/\mathbb E\|g_B\|^2<2/\eta$. Write this ratio of expectations as a weighted average of the $S_B$. When does it differ from $\mathrm{BS}=\mathbb E[S_B]$?

(c) Treat $L_B$ as exactly quadratic near $\theta$. Show that the same batch's gradient after its own step satisfies $\|\nabla L_B(\theta')\|^2/\|g_B\|^2\ge(1-\eta S_B)^2$, and deduce $\mathbb E_B\big[\|\nabla L_B(\theta')\|^2/\|g_B\|^2\big]\ge(1-\eta\,\mathrm{BS})^2$.

(d) Does (c) show that SGD is unstable whenever $\mathrm{BS}>2/\eta$? Test it on two equally likely quadratic batches $H_1=\operatorname{diag}(2.3,1)$, $H_2=\operatorname{diag}(1,2.3)$ (both with minimum at $0$), $\eta=1$, $\theta=(1,1)$: compute $\mathrm{BS}(\theta)$ and $\mathbb E_B\|\theta'\|^2$. What, then, does (c) say, and does it explain why $\mathrm{BS}$ rather than the ratio in (b) settles at $2/\eta$?

<details><summary>Hint</summary>

(c) On a quadratic, $\nabla L_B(\theta')=(I-\eta H_B)g_B$. Expand the squared norm and use Cauchy–Schwarz, $(g^\top Hg)^2\le\|g\|^2\|Hg\|^2$. Then use Jensen's inequality for the convex function $s\mapsto(1-\eta s)^2$.

(d) With both batch minima at $0$, $\theta'=(I-H_B)\theta$, so $\mathbb E\|\theta'\|^2=\theta^\top\,\mathbb E[(I-H_B)^2]\,\theta$.
</details>
"""
    ),
    answer(
        r"""
**(a)** $L_B(\theta-\eta g_B)=L_B(\theta)-\eta\|g_B\|^2+\tfrac12\eta^2g_B^\top H_Bg_B+O(\eta^3)$. The change is negative iff $\tfrac12\eta^2g_B^\top H_Bg_B<\eta\|g_B\|^2$, i.e. $S_B<2/\eta$ (for $g_B\neq0$). Each step sees its own one-dimensional quadratic along its own gradient, with curvature $S_B$: the quadratic stability condition $\eta\lambda<2$, batch by batch.

**(b)** Taking expectations of (a): $\mathbb E\,\Delta L_B=-\eta\,\mathbb E\|g_B\|^2+\tfrac12\eta^2\,\mathbb E[g_B^\top H_Bg_B]$, negative iff

$$
\frac{\mathbb E[g_B^\top H_Bg_B]}{\mathbb E\|g_B\|^2}<\frac2\eta .
$$

Since $g_B^\top H_Bg_B=S_B\|g_B\|^2$, the left side is $\mathbb E[w_BS_B]$ with weights $w_B=\|g_B\|^2/\mathbb E\|g_B\|^2$ (which average to $1$): a gradient-norm-weighted average of the $S_B$, whereas $\mathrm{BS}$ weights every batch equally. The difference is $\mathbb E[(w_B-1)S_B]=\operatorname{Cov}(w_B,S_B)$, nonzero when $S_B$ is correlated with $\|g_B\|^2$. Since the paper finds the weighted ratio above $2/\eta$ while $\mathrm{BS}\approx2/\eta$, batches with larger gradients have, on average, larger curvature along their gradient.

**(c)** On a quadratic, $\nabla L_B(\theta')=(I-\eta H_B)g_B$, so

$$
\frac{\|\nabla L_B(\theta')\|^2}{\|g_B\|^2}=1-2\eta S_B+\eta^2q_B,\qquad q_B=\frac{\|H_Bg_B\|^2}{\|g_B\|^2}\ge S_B^2
$$

by Cauchy–Schwarz. Hence the ratio is at least $(1-\eta S_B)^2$. Averaging over $B$ and applying Jensen to the convex $s\mapsto(1-\eta s)^2$:

$$
\mathbb E_B\frac{\|\nabla L_B(\theta')\|^2}{\|g_B\|^2}\ \ge\ \mathbb E_B(1-\eta S_B)^2\ \ge\ (1-\eta\,\mathbb E S_B)^2=(1-\eta\,\mathrm{BS})^2 .
$$

**(d)** No. In the example, $g_1=H_1\theta=(2.3,1)$ and $S_1=(2.3^3+1)/(2.3^2+1)\approx2.093$, and by symmetry $S_2=S_1$, so $\mathrm{BS}\approx2.093>2/\eta$. But $\mathbb E[(I-H_B)^2]=\tfrac12\big(\operatorname{diag}(1.69,0)+\operatorname{diag}(0,1.69)\big)=0.845\,I$, so $\mathbb E\|\theta'\|^2=0.845\|\theta\|^2$: SGD contracts in mean square. What (c) controls is the *same-batch* amplification: how much a step inflates the gradient of the batch it was taken on. Different batches' factors do not multiply into one quantity that tracks the iterate, so (c) is not an instability theorem. The paper's Prop. 4 shows that if batch sharpness stays above $(2+\varepsilon)/\eta$, the expected *product* of these same-batch factors over successive steps grows exponentially (App. D.1). That product is not the growth of any one gradient norm, so it is weaker than "training is unstable", as the example shows. Why training pushes $\mathrm{BS}$ *up to* $2/\eta$ and holds it there, and why the weighted ratio settles higher, are empirical findings.
"""
    ),
    markdown(
        r"""
**Code: batch sharpness in the MLP.** SGD at $\eta=0.3$ ($2/\eta\approx6.67$) on the §3.1 network, batch sizes $b=16$ and $64$. Every 100 steps: full-batch $\lambda_{\max}$, $\mathrm{BS}$, and the ratio of expectations from Exercise 3.6(b), each expectation over 32 random batches. A small, noisy run.
"""
    ),
    code(
        r"""
ALL = torch.arange(n_data)

def loss_on(net, idx):
    return 0.5 * ((net(X[idx]) - Y[idx]) ** 2).mean()

def batch_stats(net, b, nb, gen):
    # returns (mean of S_B, ratio of expectations) over nb random batches
    params = list(net.parameters()); rq, num, den = [], 0.0, 0.0
    for _ in range(nb):
        idx = torch.randperm(n_data, generator=gen)[:b]
        g = torch.autograd.grad(loss_on(net, idx), params, create_graph=True)
        gd = [t.detach() for t in g]
        Hg = torch.autograd.grad(g, params, grad_outputs=gd)
        q = sum((a * c).sum() for a, c in zip(Hg, gd)).item(); gg = sum((a * a).sum() for a in gd).item()
        rq.append(q / gg); num += q; den += gg
    return float(np.mean(rq)), num / den

def run_sgd(eta, b, steps=6000, every=100, nb=32):
    net = make_mlp(0); gen = torch.Generator().manual_seed(1); v, rec = None, []
    for k in range(steps + 1):
        if k % every == 0:
            lam, v = sharpness(net, v, iters=20 if v is None else 6)
            rec.append((k, lam, *batch_stats(net, b, nb, gen)))
        idx = torch.randperm(n_data, generator=gen)[:b]
        grads = torch.autograd.grad(loss_on(net, idx), list(net.parameters()))
        with torch.no_grad():
            for q, g in zip(net.parameters(), grads):
                q -= eta * g
    return np.array(rec)

t0 = time.time(); eta = 0.3
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for i, b in enumerate((16, 64)):
    r = run_sgd(eta, b)
    ax[i].plot(r[:, 0], r[:, 1], label="full-batch $\\lambda_{\\max}$")
    ax[i].plot(r[:, 0], r[:, 2], label="batch sharpness $\\mathbb{E}[S_B]$")
    ax[i].plot(r[:, 0], r[:, 3], lw=0.8, label="ratio of expectations")
    ax[i].axhline(2 / eta, color="k", ls="--", lw=1)
    ax[i].set(xlabel="step", title=f"SGD, b = {b} (dashed: $2/\\eta$)")
    late = r[r[:, 0] >= 3600]
    print(f"b = {b:2d}, steps 3600-6000: lambda_max {late[:, 1].mean():.2f}, batch sharpness {late[:, 2].mean():.2f} "
          f"(+-{late[:, 2].std():.2f}), ratio of expectations {late[:, 3].mean():.2f};  2/eta = {2/eta:.2f}")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()
print(f"({time.time() - t0:.1f} s)")
"""
    ),
    markdown(
        r"""
The run shows the gap clearly: $\lambda_{\max}$ is about 4.0 at $b=16$ and 6.6 at $b=64$. It cannot tell batch sharpness from the ratio of expectations (6.2 and 7.1 at $b=16$; 6.9 and 7.05 at $b=64$; fluctuations about $\pm0.5$); that needs longer, larger runs (the paper's App. J).
"""
    ),
    # ================================================================== 3.7 catapult
    markdown(
        r"""
<a id="td23-7"></a>

## 3.7 The catapult

What if the step is too large for the curvature $\lambda_0$ already at initialization, $\eta\lambda_0>2$? In a wide two-layer linear network the dynamics reduce to two exact scalar maps with three phases: *lazy* ($\eta\lambda_0<2$), *catapult* ($2<\eta\lambda_0<4$: the loss spikes, the curvature drops below $2/\eta$, and training converges somewhere flatter) and *divergent* ($\eta\lambda_0>4$).

**Paper:** Lewkowycz, Bahri, Dyer, Sohl-Dickstein & Gur-Ari, *The catapult mechanism* (see [Readings](#td23-readings)).

Like self-stabilization (§3.2), this is "oscillation squared lowers curvature", but with no sharpening pressure pushing back, so the drop is one-shot. The coupling is weak, $O(1/n)$ at width $n$, so the loss must grow to order $n$ (about $\log n$ steps) before the drop: the spike is part of the mechanism. Cohen et al. 2021 suggest EoS may be a series of micro-catapults; the first drop in the §3.1 plot resembles one. The edge $4/\lambda_0$ is specific to this model: ReLU networks tolerate up to about $12/\lambda_0$ (tanh, closer to $4/\lambda_0$).

<a id="td23-7-walkthrough"></a>

**Derivations and walkthrough.** *The model:* $f=v^\top u/\sqrt n$ with weights $u,v\in\mathbb R^n$ (here $u$ is a weight vector, not the eigenvector of earlier sections; their $f(x)=n^{-1/2}v^\top ux$ with a single input $x=1$), target $0$, loss $L=\tfrac12f^2$, and the neural tangent kernel $\lambda=\|\nabla f\|^2=(\|u\|^2+\|v\|^2)/n$, which at a minimum ($f=0$) is the sharpness. With standard initialization $u,v\sim\mathcal N(0,I)$, $f_0=O(1)$ and $\lambda_0\approx2$. One gradient step with constant $\eta$ gives, exactly,

$$
f_{k+1}=f_k\Big(1-\eta\lambda_k+\frac{\eta^2f_k^2}n\Big),\qquad \lambda_{k+1}=\lambda_k+\frac{\eta f_k^2}n\big(\eta\lambda_k-4\big).
$$

The $\lambda$-update is driven by $f^2$ as the toy's $y$ is driven by $x^2$ in §3.2. The phase boundaries are sharp as $n\to\infty$ with $\eta\lambda_0$ held away from $2$ and $4$.

"""
        + FIG_CAT
        + r"""

**Exercise 3.7 — The catapult maps.**

(a) Derive the two maps from one gradient step on $(u,v)$.

(b) Show that at $f=0$ the Hessian of $L$ is $\nabla f\,\nabla f^\top$ and its top eigenvalue is $\lambda$, so $\eta\lambda<2$ is the quadratic stability condition.

(c) Read the three phases off the maps, for large $n$ and $f_0=O(1)$. In the catapult phase, estimate how many steps the growth lasts, and say why a run that converges must end with $\eta\lambda_\infty\le2$. In the lazy phase, how far does $\lambda$ move?

<details><summary>Hint</summary>

(a) $\partial L/\partial u=f\,v/\sqrt n$ and $\partial L/\partial v=f\,u/\sqrt n$. Compute $v_{k+1}^\top u_{k+1}$ and $\|u_{k+1}\|^2+\|v_{k+1}\|^2$, using $v^\top u=\sqrt n\,f$.

(c) Write $a=\eta\lambda$ and $b=\eta^2f^2/n$, so the maps read $f'=f(1-a+b)$, $a'=a+b(a-4)$. Start with $b_0\ll1$. Watch the sign of $a-4$.
</details>
"""
    ),
    answer(
        r"""
**(a)** One step: $u'=u-\eta f\,v/\sqrt n$, $v'=v-\eta f\,u/\sqrt n$. Then

$$
v'^\top u'=v^\top u-\frac{\eta f}{\sqrt n}\big(\|u\|^2+\|v\|^2\big)+\frac{\eta^2f^2}n\,u^\top v ,
$$

and dividing by $\sqrt n$, with $v^\top u=\sqrt nf$ and $\|u\|^2+\|v\|^2=n\lambda$:

$$
f'=f-\eta f\lambda+\frac{\eta^2f^3}n=f\Big(1-\eta\lambda+\frac{\eta^2f^2}n\Big).
$$

For the kernel, $\|u'\|^2=\|u\|^2-2\eta f\,u^\top v/\sqrt n+\eta^2f^2\|v\|^2/n$ and symmetrically for $v'$. Adding, with $u^\top v/\sqrt n=f$:

$$
n\lambda'=n\lambda-4\eta f^2+\eta^2f^2\lambda\quad\Longrightarrow\quad\lambda'=\lambda+\frac{\eta f^2}n(\eta\lambda-4).
$$

(A sympy check is in `td/checks/td2_3_edge_of_stability_sympy.py`; the code below checks the maps against explicit gradient descent on $(u,v)$.)

**(b)** $\nabla^2L=\nabla f\nabla f^\top+f\,\nabla^2f$, and at $f=0$ the second term vanishes. A rank-one matrix $aa^\top$ has top eigenvalue $\|a\|^2$, here $\|\nabla f\|^2=(\|v\|^2+\|u\|^2)/n=\lambda$. So gradient descent near the minimum is stable iff $\eta\lambda<2$, and $\lambda$ is the sharpness at the minimum.

**(c)** With $a=\eta\lambda$, $b=\eta^2f^2/n$: $f'=f(1-a+b)$ and $a'=a+b(a-4)$, and $b_0=\eta^2f_0^2/n\ll1$.

*Lazy* ($a_0<2$): $|1-a+b|\approx|1-a|<1$, so $f$ decays geometrically. The total change in $a$ is $\sum_kb_k(a_k-4)$, a geometric sum of size $O(b_0)$, so $\lambda$ moves only by $O(1/n)$: the kernel is frozen and the network behaves as its linearization.

*Catapult* ($2<a_0<4$, and $a_0>2+b_0$ so that the first multiplier exceeds $1$ in magnitude): $1-a<-1$, so $|f|$ grows by about $a_0-1$ per step with alternating sign, while $a$ *decreases* (since $a-4<0$) by $b|a-4|$ per step, negligible until $b=O(1)$, i.e. $f^2\sim n/\eta^2$. Since $|f_k|\approx|f_0|(a_0-1)^k$, that takes about $\log\!\big(n/(\eta f_0)^2\big)/\big(2\log(a_0-1)\big)$ steps: logarithmic in the width. Then $a$ drops by $O(1)$ per step. A run cannot settle while $a>2$: settling means $f\to0$, hence $b\to0$, and then the multiplier $|1-a|>1$ makes $f$ grow again. Once $a<2$ with $b$ small, $f$ decays geometrically and $a$ freezes, because its updates are proportional to $f^2$. So a run that converges ends with $\eta\lambda_\infty\le2$: it lands in a flatter region than it started. (Where exactly it lands depends sensitively on $a_0$ and $f_0$; the code shows it.)

*Divergent* ($a_0>4$): $a-4>0$, so the same $f^2$ that grows also *increases* the curvature, which increases the growth rate of $f$.

Near the boundaries the finite-$n$ maps blur these statements: at $a_0=4$ exactly, $a$ never changes; and just above $a_0=2$ with small $n$, the $+b$ term can keep the first multiplier below $1$ in magnitude, so there is no spike.
"""
    ),
    code(
        r"""
def catapult(f0, a0, n, eta=1.0, K=200):
    # Iterate the exact maps; returns arrays f_k and a_k = eta*lambda_k (NaN after divergence).
    f, a = np.full(K + 1, np.nan), np.full(K + 1, np.nan)
    f[0], a[0] = f0, a0
    for k in range(K):
        b = eta**2 * f[k] ** 2 / n
        f[k + 1], a[k + 1] = f[k] * (1 - a[k] + b), a[k] + b * (a[k] - 4)
        if not np.isfinite(f[k + 1]) or abs(f[k + 1]) > 1e100:
            f[k + 1] = a[k + 1] = np.nan; break
    return f, a

# 1. the maps agree with explicit gradient descent on (u, v)
n = 1000; rng = np.random.default_rng(0)
u, v = rng.standard_normal(n), rng.standard_normal(n)
f0, lam0 = v @ u / np.sqrt(n), (u @ u + v @ v) / n
eta = 3.0 / lam0
fm, am = catapult(f0, eta * lam0, n, eta, K=40)
uu, vv, err = u.copy(), v.copy(), 0.0
for k in range(40):
    fk, ak = vv @ uu / np.sqrt(n), eta * (uu @ uu + vv @ vv) / n
    err = max(err, abs(fk - fm[k]) / (1 + abs(fm[k])), abs(ak - am[k]))
    uu, vv = uu - eta * fk * vv / np.sqrt(n), vv - eta * fk * uu / np.sqrt(n)
print(f"f0 = {f0:.3f}, lambda0 = {lam0:.3f}; max |maps - explicit GD| over 40 steps: {err:.1e}")

# 2. the three phases
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a0, col in ((1.5, "C0"), (3.0, "C1"), (4.1, "C3")):
    f, a = catapult(1.0, a0, n)
    ax[0].semilogy(0.5 * f**2, color=col, label=f"$\\eta\\lambda_0$ = {a0}")
ax[0].axhline(0.5 * n, color="gray", ls=":", lw=1)
ax[0].set(xlabel="step $k$", ylabel="loss $f^2/2$", xlim=(0, 60), ylim=(1e-12, 1e12), title=f"loss (dotted: $n/2$, n = {n})")
ax[0].legend()
grid = np.linspace(0.2, 4.4, 400)
fin, conv = [], []
for a0 in grid:
    f, a = catapult(1.0, a0, n, K=400)
    ok = np.isfinite(f[-1]) and abs(f[-1]) < 1e-6            # converged within 400 steps
    fin.append(a[-1] if np.isfinite(f[-1]) else np.nan); conv.append(ok)
fin, conv = np.array(fin), np.array(conv)
ax[1].plot(grid[conv], fin[conv], ".", ms=2.5, color="C0", label="converged")
ax[1].plot(grid[~conv], fin[~conv], "x", ms=3, color="C3", label="not converged in 400 steps")
ax[1].plot(grid, grid, color="gray", lw=0.8)
for x_, c_ in ((2, "C2"), (4, "C3")):
    ax[1].axvline(x_, color=c_, ls="--", lw=1)
ax[1].axhline(2, color="C2", ls=":", lw=1)
ax[1].set(xlabel="$\\eta\\lambda_0$", ylabel="final $\\eta\\lambda$", ylim=(0, 4.5),
          title="final curvature (gray: no change; missing: diverged)")
ax[1].legend(fontsize=7); plt.tight_layout(); plt.show()
"""
    ),
    markdown(
        r"""
**The right-hand plot** (with $a=\eta\lambda$, $b=\eta^2f^2/n$): the catapult over-flattens. For $\eta\lambda_0$ between about 3 and 4 the final $\eta\lambda$ is often far below 2, sometimes below 0.1, in lobes set by how the $O(1)$-per-step drop of $a$ happens to end. Runs landing at very small $a$ converge slowly (red crosses): $\|u\|^2+\|v\|^2\ge2|u^\top v|$ gives $a\ge2\sqrt b$, so at $a\approx0$ the multiplier $1-a+b$ is close to $1$. They are simply very flat, not near a phase boundary.
"""
    ),
    # ================================================================== 3.8
    markdown(
        r"""
<a id="td23-8"></a>

## 3.8 Warmup and WSD

Why does ramping the learning rate up at the start (*warmup*) help, and why does the loss drop sharply in the final decay of a warmup–stable–decay (WSD) schedule? Both have readings in terms of oscillation across sharp directions.

**Papers:** Kalra & Barkeshli, *Why warmup the learning rate?*; Wen et al., *Understanding warmup-stable-decay learning rates* (see [Readings](#td23-readings)).

*Warmup:* a large step from a sharp start throws the network into a catapult (§3.7) or divergence; ramping $\eta$ up lets the sharpness follow $2/\eta$ down to where the target step is tolerable. *WSD:* the loss is a *river valley*, a flat river between steep hillsides. At a large step the iterate bounces between the hillsides while progressing along the river; the bouncing hides the progress, and the decay removes it.

<a id="td23-8-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* $\eta_k$ is the step at step $k$, $\eta_{\rm trgt}$ the target step after warmup, $S$ the sharpness (top Hessian eigenvalue), and time $t=\sum_{j<k}\eta_j$.

**Warmup.** Kalra and Barkeshli use the linear version of the §3.2 model: $x_{k+1}=(1-\eta_k\lambda_k)x_k$, $\lambda_{k+1}=\lambda_k+\eta_k(\alpha-\beta x_k^2)$, with $x$ the displacement along the top eigenvector, $\lambda$ the sharpness, $\alpha>0$ a sharpening rate and $\beta>0$ the strength of the $x^2$ kick. Exercise 3.8(a) runs the experiment in our toy. (For Adam an earlier, compatible explanation is that its second-moment average starts from few gradients, so the adaptive step has high variance until warmup lets it settle: [Liu et al.'s RAdam paper (ICLR 2020)](https://arxiv.org/abs/1908.03265).)

**WSD.** In Wen et al. the bouncing comes from SGD noise (Exercise 3.8(b)); with full-batch small-step gradient descent their iterates track the river. A deterministic analogue, from §3.3 (a connection neither paper makes): at the edge the measured loss is $L(w)+\sigma^2/\eta$ ($w$ the averaged iterate, $\sigma^2$ the oscillation variance), and decaying $\eta$ raises the edge $2/\eta$, so the oscillation and its excess loss die out until sharpening reaches the new edge. For the SGD side, see [NB2 §2](td2_2_sgd_sde.html).

**Exercise 3.8 — Warmup in the toy, and a river valley.**

**(a) Code.** Start the §3.2 toy ($L=\tfrac12yx^2+c\,y^{-5}$, $c=62.5$, sharpness $S=y$) sharp, at $y_0=8$, and train at $\eta_{\rm trgt}=0.4$, so $2/\eta_{\rm trgt}=5$. Compare jumping straight to $\eta_{\rm trgt}$ with linear warmups $\eta_k=\eta_{\rm trgt}\min\big(1,(k+1)/T\big)$ for $T\in\{100,400,1600\}$, using `toy_run` with 4000 steps. For each, plot $S_k=y_k$ and $2/\eta_k$ against $t$, and report whether it diverged and its peak loss. Then repeat from $y_0=5.5$: when does warmup help, and can it hurt? Panel 1 of the interactive figure in §3.2 has the same experiment with sliders.

**(b) Derivation.** Take $L(h,r)=\tfrac12\gamma h^2-r$: a hill coordinate $h$ with curvature $\gamma$ and a river coordinate $r$ along which the loss falls at unit slope. Run SGD with noise only in the hill direction: $h_{k+1}=h_k-\eta\gamma h_k+\eta\xi_k$, $\xi_k\sim\mathcal N(0,s^2)$ i.i.d. (so $s^2$ is the noise variance), and $r_{k+1}=r_k+\eta$. Find the stationary mean of the hill loss $\tfrac12\gamma h^2$ at constant $\eta$, for $0<\eta\gamma<2$ (exactly, and for $\eta\gamma\ll1$). Then explain what a WSD schedule (long constant phase at $\eta_{\max}$, short decay to $\eta_{\min}$) achieves compared with (i) $\eta_{\max}$ throughout, (ii) $\eta_{\min}$ throughout.

<details><summary>Hint</summary>

(b) Take the variance of both sides of the $h$-update at stationarity. The river coordinate after $k$ steps is $r_0+\sum_j\eta_j$, which in our time variable is just $t$. After $\eta$ changes, how does the variance $v_k=\operatorname{Var}h_k$ evolve, and how long does it take to reach its new stationary value?
</details>
"""
    ),
    code_answer(
        r"""
# Exercise 3.8(a): warmup in the toy. Use toy_run and toy_loss from §3.2.
eta_trgt, K, y0 = 0.4, 4000, 8.0
schedules = {"no warmup": np.full(K, eta_trgt)}
# add linear warmups for T in (100, 400, 1600), run each, and plot y_k and 2/eta_k against t = cumsum(eta)
""",
        r"""
eta_trgt, K = 0.4, 4000
for y0 in (8.0, 5.5):
    schedules = {"no warmup": np.full(K, eta_trgt)}
    for T in (100, 400, 1600):
        schedules[f"warmup T={T}"] = eta_trgt * np.minimum(1, (np.arange(K) + 1) / T)
    fig, ax = plt.subplots(1, len(schedules), figsize=(13, 2.8), sharey=True)
    for i, (name, etas) in enumerate(schedules.items()):
        xs, ys = toy_run(etas, y0=y0)
        t = np.concatenate([[0], np.cumsum(etas)[:-1]])
        diverged = np.isnan(ys).any()
        peak = np.nanmax(toy_loss(xs, ys))
        ax[i].plot(t, ys, lw=0.8, label="$S_k$")
        ax[i].plot(t, 2 / etas, "--", lw=1, label="$2/\\eta_k$")
        ax[i].set(title=f"{name}: " + ("DIVERGED" if diverged else f"peak loss {peak:.2g}"),
                  xlabel="$t$", ylim=(0, 10))
        kc = np.argmax(etas * ys > 2)                                  # first crossing of the edge
        over = np.nanmax((etas * ys)[kc:kc + 200])
        print(f"y0 = {y0}, {name:16s}: " + ("diverged" if diverged else
              f"peak loss = {peak:.3g}, S at crossing = {ys[kc]:.2f}, max eta*S after crossing = {over:.2f}, "
              f"final S = {ys[-1]:.3f} (2/eta = {2/eta_trgt})"))
    ax[0].set_ylabel(f"$y_0$ = {y0}"); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.show()
"""
    ),
    answer(
        r"""
**(a)** From $y_0=8$ with $\eta=0.4$, $\eta S_0=3.2$: the oscillation grows by a factor of about $2.2$ per step, its $x^2$ kick throws $y$ far below $2/\eta$ and through zero, and the run diverges. A short warmup ($T=100$) crosses the edge before the sharpness can move, and diverges here (it is near the survival boundary: with other noise seeds some runs survive with a large spike). Longer warmups let the sharpness follow $2/\eta(t)$ down, not smoothly: from the edge (at $\eta\approx2/8$) on, $S$ circles $2/\eta(t)$ in small catapults, because the target moves faster than the slow attraction of the period-2 orbit (Exercise 3.2′(f)); this is Kalra and Barkeshli's persistent cycle. The peak loss falls as $T$ grows (about 15 at $T=400$, 3.6 at $T=1600$), and the run ends at $S=2/\eta_{\rm trgt}=5$.

From $y_0=5.5$ ($\eta S_0=2.2$) the run survives without warmup, with a peak loss of about $2.7$. Short warmups do not help here and can hurt (about $3.8$ at $T=100$, $3.5$ at $T=400$); only $T=1600$ lowers the peak (about $1.6$). Two things set the spike size: how far $\eta S$ overshoots 2 before the $x^2$ kick arrives (printed max $\eta S$ after the crossing: about 2.2–2.3 for no or short warmup, 3.1 for $T=1600$), and the sharpness at the crossing, which is higher after a warmup because the toy keeps sharpening while $\eta$ is small ($\ell'<0$ never switches off). A slow ramp crosses gently; a short one adds sharpening without the gentleness. So warmup pays when $\eta_{\rm trgt}S_0$ is well above 2. In the toy, what moves the network to where $\eta_{\rm trgt}$ is tolerable is the edge of stability itself. (Numbers are for the displayed noise seed.)

**(b)** At stationarity $\operatorname{Var}h=(1-\eta\gamma)^2\operatorname{Var}h+\eta^2s^2$, so for $0<\eta\gamma<2$

$$
\operatorname{Var}h=\frac{\eta^2s^2}{1-(1-\eta\gamma)^2}=\frac{\eta s^2}{\gamma(2-\eta\gamma)},\qquad
\mathbb E\big[\tfrac12\gamma h^2\big]=\frac{\eta s^2}{2(2-\eta\gamma)}\ \approx\ \frac{\eta s^2}4\quad(\eta\gamma\ll1).
$$

The excess hill loss is proportional to $\eta$ and, for small $\eta\gamma$, independent of the curvature $\gamma$: every noisy steep direction contributes about $\eta s^2/4$. Meanwhile the river coordinate after $k$ steps is $r_0+\sum_j\eta_j=r_0+t$: river progress is set by the *area under the schedule*. So the expected loss is $-t+\eta s^2/4$ (plus a constant), once the hill has equilibrated.

(i) Constant $\eta_{\max}$: fast river progress, but the hill term $\eta_{\max}s^2/4$ never goes away and hides part of it. (ii) Constant $\eta_{\min}$: small hill term, but river progress per step is smaller by $\eta_{\max}/\eta_{\min}$. WSD gets both: the long stable phase accumulates nearly all the area, and the decay lowers the hill term. The decay is not instantaneous: $v_{k+1}=(1-\eta_k\gamma)^2v_k+\eta_k^2s^2$, so the old variance takes about $1/(2\eta\gamma)$ steps per e-fold at the current $\eta$, and a decay to a very small $\eta_{\min}$ must be long enough (in $\sum\eta\gamma$) for the hill to follow. The loss drop during the decay is the hill term disappearing, revealing progress made earlier. This is Wen et al.'s mechanism; their theorems cover more general valleys, and their constants are not reproduced here.
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "td2_3_edge_of_stability", "td2_3_edge_of_stability"))
