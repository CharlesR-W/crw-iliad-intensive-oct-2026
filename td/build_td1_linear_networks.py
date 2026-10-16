#!/usr/bin/env python3
"""Build NB1 of the Training Dynamics day: Saxe dynamics and linearity.

Deep linear networks: mode ODEs, depth, landscape, the end-to-end NTK, the
rich / lazy / mixed regimes, and which Saxe-derived heuristics are derived
versus planted. Closed forms are checked in checks/td1_linear_networks_math.py;
the interactive figure is checked by assets/interactive/checks/mode_explorer_check.py.
Review log: ../reviews/td/td1_linear_networks.md.

Page shape (readings first, 2026-10-14): a short intro and a one-line route, then one
`## Readings` section (anchor td1-readings) with every paper of the notebook, grouped by
section, then the derivations and walkthrough: six `##` sections (anchors td1-s1 ..
td1-s6), each with the question and its answer in plain words and at most a short
core-idea paragraph, then (from the td1-sN-walkthrough anchor) the derivations,
exercises with worked solutions and code; a short closing `## 7. Going further`
(td1-s7) keeps the pointers to the Physics of Deep Learning day and to NB2. Sub-topics
keep their old numbers as bold labels (**1.4 ...**), so references like "§1.4" and
"Exercise 1.4" still make sense. Exercises are bold paragraphs with anchors td1-exN-M.
No `###` or `####`.

Time convention (NB1 only): tau = 1/eta, so t counts optimizer steps and
t/tau = eta * steps.
"""
from nbkit_td import answer, book, code, code_answer, figure, markdown, reading, references, run

PHYS = "../../IliadPhysicsLearning/notebooks/second-draft"

SAXE14 = ("Exact solutions to the nonlinear dynamics of learning in deep linear neural networks",
          "https://arxiv.org/abs/1312.6120", "Saxe, McClelland, Ganguli (ICLR 2014)")
TAJ = ("Mixed dynamics in linear networks: unifying the lazy and active regimes",
       "https://arxiv.org/abs/2405.17580", "Tu, Aranguri, Jacot (NeurIPS 2024)")

CELLS = [
    # ================================================================== intro
    markdown(
        r"""
# NB1 — Saxe dynamics and linearity

**{{VARIANT}} · Training Dynamics, Iliad Intensive 2026**

A deep linear network can only represent a linear map, but gradient descent on it is
nonlinear and exactly solvable. Long plateaus followed by sudden drops, the strongest
structure learned first, saddle-to-saddle trajectories, and a tangent kernel that either
moves or stays frozen all appear here in closed form.

**Route.** The [Readings](#td1-readings) come first: every paper of the notebook, grouped
by section. The sections after them are the derivations and walkthrough (exercises with
worked solutions, and code), for whoever wants them.

- [§1 The mode ODEs](#td1-s1): one solvable ODE per singular mode of the data.
- [§2 Depth](#td1-s2) (side branch): plateaus grow as a power of the initial scale.
- [§3 The landscape](#td1-s3) (side branch): no bad local minima, which says little about training time.
- [§4 The end-to-end flow and its kernel](#td1-s4): the network's NTK and one conserved number per mode.
- [§5 Rich, lazy and mixed](#td1-s5): one dial; large balanced initialization is not lazy.
- [§6 Derived or planted](#td1-s6): which Saxe-style heuristics come from the math, and which from the setup.
- [§7 Going further](#td1-s7): the Physics of Deep Learning day's kernel notebooks, and where NB2 picks this up.

The code cells share these helpers (numpy and matplotlib):
"""
    ),
    code(
        r"""
# Shared helpers (run this first). numpy + matplotlib only.
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (8, 3.4), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

def rk4(f, y0, t_grid, substeps=20):
    # Integrate y' = f(y) with classical RK4; return y at every time in t_grid.
    y = np.array(y0, dtype=float); out = [y.copy()]
    for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
        h = (t1 - t0) / substeps
        for _ in range(substeps):
            k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y.copy())
    return np.array(out)

def make_target(s, n=6, seed=0):
    # Sigma = U diag(s) V^T in n dimensions, with random orthogonal U, V.
    r = np.random.default_rng(seed)
    U, _ = np.linalg.qr(r.normal(size=(n, n))); V, _ = np.linalg.qr(r.normal(size=(n, n)))
    k = len(s)
    return U[:, :k] @ np.diag(s) @ V[:, :k].T, U, V

def product(Ws):
    # End-to-end map W_N ... W_1 of a list [W_1, ..., W_N].
    W = Ws[0]
    for M in Ws[1:]:
        W = M @ W
    return W

def gd(Ws, Sigma, eta, steps, every=10):
    # Full-batch GD on L = 1/2 ||Sigma - W_N...W_1||_F^2 + const (whitened inputs).
    # Returns the times t/tau = eta*k and the weights [W_1, ..., W_N] at those times.
    Ws = [M.copy() for M in Ws]; ts, hist = [], []
    for k in range(steps + 1):
        if k % every == 0:
            ts.append(eta * k); hist.append([M.copy() for M in Ws])
        if k == steps:
            break
        E = Sigma - product(Ws)
        grads = []
        for j in range(len(Ws)):
            left = product(Ws[j + 1:]) if j + 1 < len(Ws) else np.eye(E.shape[0])
            right = product(Ws[:j]) if j > 0 else np.eye(E.shape[1])
            grads.append(left.T @ E @ right.T)           # this is -dL/dW_j
        Ws = [M + eta * g for M, g in zip(Ws, grads)]
    return np.array(ts), hist

def mode_strengths(W, U, V, k):
    # a_alpha = u_alpha^T W v_alpha for the first k modes.
    return np.diag(U[:, :k].T @ W @ V[:, :k])

def logistic(t, s, a0):
    # Balanced two-layer solution, t in units of tau.
    e = np.exp(np.minimum(2 * s * t, 700))
    return s * e / (e - 1 + s / a0)

def t_of_a(a, s, d, a0):
    # Closed-form time (units of tau) for tau a' = sqrt(4a^2 + d^2)(s - a) to go from a0 to a.
    R = np.sqrt(4 * s**2 + d**2)
    F = lambda x: np.log((4 * s * x + d**2 + R * np.sqrt(4 * x**2 + d**2)) / (2 * s - 2 * x)) / R
    return F(a) - F(a0)

def mode_ode(s, Delta, a0, t_grid):
    # Numerical solution of tau a' = sqrt(4a^2 + Delta^2)(s - a).
    return rk4(lambda a: np.sqrt(4 * a**2 + Delta**2) * (s - a), [a0], t_grid)[:, 0]
"""
    ),
    # ================================================================== readings
    markdown(
        r"""
<a id="td1-readings"></a>

## Readings

**The notebook as a whole.** The exact solutions ([§1](#td1-s1), [§2](#td1-s2)) and the
lazy / rich / mixed picture ([§4](#td1-s4), [§5](#td1-s5)).
"""
        + reading(
            *SAXE14,
            "The exact solutions this notebook is built on: decoupled modes, sigmoidal learning curves with plateaus, and what depth changes. Sections 1–3 derive the decoupled mode equations and their logistic solutions (§1); Eq. 15 and App. B give §2's depth-$N$ equation.",
            "Their $\\Sigma$'s are sums over examples rather than averages, and they count layers including the input, so their $N_l$ is our $N+1$.",
            importance=5,
            length=3,
        )
        + reading(
            *TAJ,
            "One formula for wide two-layer linear networks that covers lazy, rich and mixed learning (§4–§5), with a phase diagram in width and initialization scale: with width $w$ and initialization variance $\\sigma^2$, singular values below about $\\sigma^2w$ are learned lazily and larger ones actively.",
            importance=5,
            length=3,
        )
        + r"""
**The mode ODEs and alignment ([§1](#td1-s1)).**
"""
        + reading(
            "Neural networks as kernel learners: the silent alignment effect",
            "https://arxiv.org/abs/2111.00034",
            "Atanasov, Bordelon, Pehlevan (ICLR 2022)",
            "Why the hidden layer lines up with the data's modes before the loss moves: from small initialization the tangent kernel aligns while still small (proved for scalar-output linear networks), so the trained network is close to kernel regression with its *final* kernel.",
            importance=4,
            length=3,
        )
        + r"""
**Depth ([§2](#td1-s2)).**
"""
        + reading(
            "Saddle-to-saddle dynamics in deep linear networks: small initialization training, symmetry, and sparsity",
            "https://arxiv.org/abs/2106.15933",
            "Jacot, Ged, Şimşek, Hongler, Gabriel (2021)",
            "Proves that from a small random initialization of per-layer scale $\\alpha$ the first escape from the origin follows a rank-one path, taking time $\\propto-\\log\\alpha$ at depth 2 and $\\alpha^{-(N-2)}$ beyond. The full saddle-to-saddle sequence is a conjecture supported by numerics.",
            importance=3,
            length=3,
        )
        + r"""
**The landscape ([§3](#td1-s3)).**
"""
        + reading(
            "The loss landscape of deep linear neural networks: a second-order analysis",
            "https://arxiv.org/abs/2107.13289",
            "Achour, Malgouyres, Gerchinovitz (JMLR 2024)",
            "Characterizes every critical point (Prop. 1–2) and classifies them into global minima, strict saddles and non-strict saddles (Theorem 7).",
            "Their depth $H$ counts weight matrices. They do not assume whitening.",
            importance=5,
            length=3,
        )
        + reading(
            "A convergence analysis of gradient descent for deep linear neural networks",
            "https://arxiv.org/abs/1810.02281",
            "Arora, Cohen, Golowich, Hu (ICLR 2019)",
            "The kind of convergence theorem the landscape alone does not give: a linear rate, given width, approximately balanced initialization, low initial loss and a small step.",
            importance=3,
            length=3,
        )
        + r"""
**The end-to-end flow ([§4](#td1-s4)).**
"""
        + reading(
            "On the optimization of deep networks: implicit acceleration by overparameterization",
            "https://arxiv.org/abs/1802.06509",
            "Arora, Cohen, Hazan (ICML 2018)",
            "The balanced end-to-end equation at any depth: a preconditioner favouring directions already large in $W$, and for $N>2$ not the gradient flow of any regularized loss.",
            importance=3,
            length=3,
        )
        + r"""
**Rich, lazy and mixed ([§5](#td1-s5)).**
"""
        + reading(
            "On lazy training in differentiable programming",
            "https://arxiv.org/abs/1812.07956",
            "Chizat, Oyallon, Bach (NeurIPS 2019)",
            "Lazy training comes from a choice of scaling (such as a large output multiplier), not from overparameterization as such.",
            importance=3,
            length=3,
        )
        + r"""
**Derived or planted ([§6](#td1-s6)).**
"""
        + reading(
            "A mathematical theory of semantic development in deep neural networks",
            "https://arxiv.org/abs/1810.10531",
            "Saxe, McClelland, Ganguli (PNAS 2019)",
            "The source of heuristic 2: the exact solutions applied to tree-structured data, giving coarse-to-fine differentiation, illusory correlations and category coherence.",
            importance=4,
            length=3,
        )
        + r"""
**Beyond linear networks ([§7](#td1-s7)).** Two extensions of the mode picture.
"""
        + reading(
            "The neural race reduction: dynamics of abstraction in gated networks",
            "https://arxiv.org/abs/2207.10430",
            "Saxe, Sodhani, Lewallen (ICML 2022)",
            "Gated linear pathways compute nonlinear functions, and their learning reduces to mode ODEs like §1's: a race between pathways.",
            importance=3,
            length=3,
        )
        + reading(
            "Alternating gradient flows: a theory of feature learning in two-layer neural networks",
            "https://arxiv.org/abs/2506.06489",
            "Kunin, Marchetti, Chen, Karkada, Simon, DeWeese, Ganguli, Miolane (NeurIPS 2025)",
            "Small-initialization training of two-layer networks as alternating steps, each neuron activation bringing a loss drop: saddle-to-saddle learning beyond linear networks.",
            importance=3,
            length=3,
        )
        + r"""
**Also:**

- [§3](#td1-s3): Baldi, Hornik, [Neural networks and principal component analysis: learning from examples without local minima](https://doi.org/10.1016/0893-6080(89)90014-2) (1989; the two-layer case, PCA for an autoencoder); Kawaguchi, [Deep learning without poor local minima](https://arxiv.org/abs/1605.07110) (2016; any depth and width, degenerate saddles from two hidden layers).
- [§5](#td1-s5): Li, Luo, Lyu, [Towards resolving the implicit bias of gradient descent for matrix factorization: greedy low-rank learning](https://arxiv.org/abs/2012.09839) (2021); Kumar, Bordelon, Gershman, Pehlevan, [Grokking as the transition from lazy to rich training dynamics](https://arxiv.org/abs/2310.06110) (2023).
- [§6](#td1-s6): Pennington, Schoenholz, Ganguli, [Resurrecting the sigmoid in deep learning through dynamical isometry](https://arxiv.org/abs/1711.04735) (2017; heuristic 4); Advani, Saxe, Sompolinsky, [High-dimensional dynamics of generalization error in neural networks](https://doi.org/10.1016/j.neunet.2020.08.022) (2020; heuristic 5; the arXiv version, [1710.03667](https://arxiv.org/abs/1710.03667), lists Advani and Saxe only).
- Wide networks beyond kernels ([§7](#td1-s7)): Mei, Montanari, Nguyen, [A mean field view of the landscape of two-layer neural networks](https://arxiv.org/abs/1804.06561) (2018); Yang, Hu, [Tensor Programs IV: feature learning in infinite-width neural networks](https://arxiv.org/abs/2011.14522) (2021; muP); Bordelon, Pehlevan, [Self-consistent dynamical field theory of kernel evolution in wide neural networks](https://arxiv.org/abs/2205.09653) (2022).
"""
        + references(
            book(
                "The Principles of Deep Learning Theory",
                "https://arxiv.org/abs/2106.10165",
                "Roberts, Yaida, Hanin (Cambridge UP 2022)",
                "finite width as a $1/n$ expansion around infinite width",
            )
        )
    ),
    # ================================================================== §1
    markdown(
        r"""
<a id="td1-s1"></a>

## 1. The mode ODEs

Train a two-layer linear network $W=W_2W_1$ on whitened inputs with square loss, and
write the input–output correlation as $\Sigma=\mathbb E[yx^\top]=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$
(its SVD). Under gradient flow, each mode's strength $a=u^\top Wv$ obeys one scalar ODE,
$\tau\dot a=\sqrt{4a^2+d^2}\,(s-a)$, where $d$ is a conserved imbalance between the two
layers and $\tau=1/\eta$ (so $t$ counts steps). Small balanced weights give a sigmoid
after a plateau whose length is set by $s$, so strong modes are learned first. Large
imbalance gives a plain exponential at a rate that does not depend on $s$.

**Papers:** Saxe, McClelland & Ganguli (2014), Sections 1–3; Atanasov, Bordelon & Pehlevan
(2022) on silent alignment (see [Readings](#td1-readings)).
"""
        + r"""
**The core idea.** Rescaling one layer up and the next down leaves $W$ unchanged; this
symmetry conserves the imbalance $D=W_2^\top W_2-W_1W_1^\top$. If the hidden layer is aligned
with the singular vectors of $\Sigma$, the flow splits into independent two-weight problems,
one per mode. Small random weights produce that alignment themselves, before the loss moves.
"""
    ),
    markdown(
        r"""
"""
        + figure(
            "assets/interactive/mode_explorer.html",
            "Sliders for $s$, the initial strength $a_0$, the imbalance $d$ and the depth $N$, with "
            "up to five modes overlaid. A second panel plots each mode's kernel $\\sqrt{4a^2+d^2}$ (and "
            "its depth-$N$ analogue), so you can see a frozen kernel versus a growing one. Presets "
            "include \"large balanced init\", which is not lazy.",
        )
    ),
    markdown(
        r"""
<a id="td1-s1-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* whitened inputs $x\in\mathbb R^{n_x}$ ($\mathbb E[xx^\top]=I$), targets
$y\in\mathbb R^{n_y}$; layers $W_1$ ($h\times n_x$) and $W_2$ ($n_y\times h$), $W=W_2W_1$; loss
$L=\tfrac12\mathbb E\|y-Wx\|^2$; $\Sigma=\mathbb E[yx^\top]=USV^\top=\sum_{\alpha=1}^rs_\alpha u_\alpha v_\alpha^\top$
with distinct $s_1>\dots>s_r>0$; error $E:=\Sigma-W$; mode strength $a_\alpha:=u_\alpha^\top Wv_\alpha$.
Gradient descent with step $\eta$ is the flow $\tau\dot\theta=-\nabla L$ with $\tau:=1/\eta$, so $t$
counts steps (Saxe et al.'s convention; NB2 uses $t=k\eta$).

**1.1 Gradient flow.** With whitened inputs $L=\tfrac12\|\Sigma-W\|_F^2+\text{const}$ (Exercise 1.1), so

$$
\tau\dot W_1=W_2^\top E,\qquad \tau\dot W_2=E\,W_1^\top,\qquad E=\Sigma-W_2W_1. \tag{1.1}
$$

The loss depends on $W$ alone, but factorizing changes the geometry in which $W$ descends (§4).

**1.2 A conserved quantity.** For any invertible $G$, $(W_1,W_2)\mapsto(GW_1,\,W_2G^{-1})$ leaves
$L$ unchanged. Its symmetric generators (rescalings and shears of the hidden layer) give
conserved charges, the entries of the **imbalance**

$$
D:=W_2^\top W_2-W_1W_1^\top\qquad(h\times h);
$$

rotations give nothing (Exercise 1.2). Finite steps break the conservation at order $\eta^2$
per step. "Balanced" ($D=0$) says nothing about the *size* of the weights, which matters in §5.

**1.3 Decoupled modes.** Suppose that at $t=0$

$$
W_1=R\,\operatorname{diag}(b)\,V^\top,\qquad W_2=U\operatorname{diag}(c)\,R^\top,
$$

with $R$ orthogonal (take $h=n_x=n_y=r$). Mode $\alpha$ then uses one hidden direction $r_\alpha$
(column $\alpha$ of $R$), read in from $v_\alpha$ with weight $b_\alpha$ and written out to $u_\alpha$
with weight $c_\alpha$. The flow preserves this form (Exercise 1.3), and each mode is gradient
flow on its own loss $\tfrac12(s-bc)^2$:

$$
\tau\dot b_\alpha=c_\alpha\,(s_\alpha-b_\alpha c_\alpha),\qquad
\tau\dot c_\alpha=b_\alpha\,(s_\alpha-b_\alpha c_\alpha). \tag{1.3}
$$

**1.4 One scalar ODE.** Drop the index. With $a=bc$ and $d=c^2-b^2$ (a diagonal entry of $D$,
so conserved), $b^2+c^2=\sqrt{4a^2+d^2}$ and

$$
\boxed{\;\tau\dot a=\sqrt{4a^2+d^2}\;(s-a)\;} \tag{1.4}
$$

The prefactor is the mode's tangent kernel (§4). Three cases (Exercise 1.4):

- **Balanced, $d=0$:** $\tau\dot a=2a(s-a)$, a logistic. From $a_0=a(0)>0$,
  $$a(t)=\frac{s\,e^{2st/\tau}}{e^{2st/\tau}-1+s/a_0},\qquad t_{1/2}=\frac{\tau}{2s}\ln\!\Big(\frac{s}{a_0}-1\Big)\approx\frac{\tau}{2s}\ln\frac{s}{a_0}.$$
  A plateau of about $(\tau/2s)\ln(s/a_0)$, then a rise in time $\sim\tau/s$. With a common small $a_0$, modes are learned in order of decreasing $s$: Saxe et al.'s sigmoids.
- **Strongly imbalanced, $|d|\gg s$:** $\tau\dot a\approx|d|(s-a)$, an exponential at rate $|d|/\tau$, the *same for every mode*. No plateau, no ordering.
- **Shallow:** one matrix trained directly has $\tau\dot a=s-a$. Linear regression has no plateaus.

So imbalance relative to $s$ (not the size of the weights, §5.3) turns the sigmoidal *rich*
behaviour into linear-model *lazy* behaviour. For general $d$ (differentiate to check; valid
for $a_0,a<s$, and $a_0>0$ if $d=0$):

$$
\frac{t}{\tau}=F(a)-F(a_0),\qquad
F(x)=\frac{1}{\sqrt{4s^2+d^2}}\,\ln\frac{4sx+d^2+\sqrt{4s^2+d^2}\,\sqrt{4x^2+d^2}}{2s-2x}.
$$

**1.5 Where alignment comes from.** Let $p_\alpha:=W_1v_\alpha$ and $q_\alpha:=W_2^\top u_\alpha$ (how
the hidden layer reads input direction $v_\alpha$ and writes output direction $u_\alpha$), so
$a_\alpha=p_\alpha\cdot q_\alpha$; aligned means both lie along one hidden direction. While $W$ is
small, $E\approx\Sigma$ and (1.1) is linear. With $Z=[W_1,\;W_2^\top]\in\mathbb R^{h\times(n_x+n_y)}$
(each hidden unit's input and output weights in one row),

$$
\tau\dot Z=Z\,M,\qquad M=\begin{pmatrix}0&\Sigma^\top\\ \Sigma&0\end{pmatrix}.
$$

$M$ has eigenvalues $\pm s_\alpha$ with eigenvectors $e^\pm_\alpha=(v_\alpha,\pm u_\alpha)/\sqrt2$, so the
$e^+_\alpha$ components grow like $e^{s_\alpha t/\tau}$ and the $e^-_\alpha$ ones decay. With
$z_\alpha:=Z(0)\,e^+_\alpha$ (Exercise 1.6),

$$
p_\alpha(t)\approx q_\alpha(t)\approx\frac{e^{s_\alpha t/\tau}}{\sqrt2}\,z_\alpha,\qquad
u_\alpha^\top W(t)\,v_\beta\approx\tfrac12\,(z_\alpha\cdot z_\beta)\,e^{(s_\alpha+s_\beta)t/\tau}.
$$

Input and output vectors become parallel and equal in length. Diagonal entries grow at rate
$2s_\alpha$ and off-diagonal ones at $s_\alpha+s_\beta$, so when the loss starts to move, $W$ is
dominated by its top mode, with effective initial strength $a_{0,1}=\tfrac12|z_1|^2$. Not
derived: that later modes separate cleanly once mode 1 is learned and then follow Saxe's
logistics. The alignment cell below and §5's code check this numerically.

<a id="td1-ex1-1"></a>

**Exercise 1.1 — Whitened loss and the flow equations.**

1. Show that $\tfrac12\mathbb E\|y-Wx\|^2=\tfrac12\|\Sigma-W\|_F^2+\tfrac12\big(\mathbb E\|y\|^2-\|\Sigma\|_F^2\big)$ when $\mathbb E[xx^\top]=I$.
2. Derive (1.1).
3. Write one step of gradient descent with step $\eta$ and check that it is the Euler discretization of (1.1) with unit time step when $\tau=1/\eta$.

<details><summary>Hint</summary>

Expand the square and use $\mathbb E[x^\top W^\top Wx]=\operatorname{tr}(W^\top W\,\mathbb E[xx^\top])$. For 2, perturb $W_1\to W_1+\delta$ and read off the linear term.
</details>
"""
    ),
    answer(
        r"""
**1.** Expanding,
$$
\tfrac12\mathbb E\|y-Wx\|^2=\tfrac12\mathbb E\|y\|^2-\mathbb E[y^\top Wx]+\tfrac12\mathbb E[x^\top W^\top Wx].
$$
The middle term is $\operatorname{tr}(W\,\mathbb E[xy^\top])=\operatorname{tr}(W\Sigma^\top)=\langle\Sigma,W\rangle_F$, and the last is $\tfrac12\operatorname{tr}(W^\top W)=\tfrac12\|W\|_F^2$ because $\mathbb E[xx^\top]=I$. Completing the square, $-\langle\Sigma,W\rangle+\tfrac12\|W\|^2=\tfrac12\|\Sigma-W\|^2-\tfrac12\|\Sigma\|^2$. The data enter only through $\Sigma$. A finite sample with $\tfrac1n\sum_ix_ix_i^\top=I$ gives the same formula with empirical averages.

**2.** $\nabla_WL=-(\Sigma-W)=-E$. Perturbing $W_1\to W_1+\delta$ changes $W$ by $W_2\delta$, so $\delta L=\langle -E,\,W_2\delta\rangle=\langle -W_2^\top E,\,\delta\rangle$ and $\nabla_{W_1}L=-W_2^\top E$. Likewise $W_2\to W_2+\delta$ changes $W$ by $\delta W_1$, and $\delta L=\langle -EW_1^\top,\delta\rangle$. With $\tau\dot\theta=-\nabla L$ this is (1.1).

**3.** One step is $W_1\leftarrow W_1+\eta\,W_2^\top E$ and $W_2\leftarrow W_2+\eta\,EW_1^\top$. Dividing by $\eta=1/\tau$ gives $\tau\,\Delta W_1=W_2^\top E$ with $\Delta$ the change per step: the Euler step of (1.1) with $\Delta t=1$. So $t$ counts steps. The continuous description is good when $\eta\times(\text{largest rate})\ll1$; the rates are computed in §2 and §4.
"""
    ),
    markdown(
        r"""
<a id="td1-ex1-2"></a>

**Exercise 1.2 — Conservation, Noether, and its breaking by finite steps.**

1. Show directly from (1.1) that $\dot D=0$.
2. *(Optional.)* Noether's argument. Take $G=e^{\epsilon A}$ with $A$ any $h\times h$ matrix. Show that invariance of $L$ implies $W_1(\nabla_{W_1}L)^\top=(\nabla_{W_2}L)^\top W_2$, and deduce $\dot D=0$ from this identity alone, without using the form of $L$. Which generators give conserved quantities, and which give none?
3. One step of gradient descent changes $D$ by $\eta^2\big(W_1E^\top EW_1^\top-W_2^\top EE^\top W_2\big)$. Verify this. Over the $O(1/\eta)$ steps needed to train, how large does the drift get?

<details><summary>Hint</summary>

For 2, differentiate $L(e^{\epsilon A}W_1,\,W_2e^{-\epsilon A})$ at $\epsilon=0$ and use $\operatorname{tr}(A M)=0$ for all $A$ $\Rightarrow M=0$. For "which generators", compute $\tfrac{d}{dt}\operatorname{tr}(A\,D)$ and note that the action $(W_1,W_2)\mapsto(AW_1,-W_2A)$ is a symmetric linear map on parameter space only when $A=A^\top$.
</details>
"""
    ),
    answer(
        r"""
**1.** Using (1.1),
$$
\tau\tfrac{d}{dt}(W_2^\top W_2)=(EW_1^\top)^\top W_2+W_2^\top EW_1^\top=W_1E^\top W_2+W_2^\top EW_1^\top,
$$
$$
\tau\tfrac{d}{dt}(W_1W_1^\top)=W_2^\top EW_1^\top+W_1(W_2^\top E)^\top=W_2^\top EW_1^\top+W_1E^\top W_2 .
$$
They are equal, so $\dot D=0$.

**2.** Write $\nabla_1,\nabla_2$ for the two gradients. Since $L(e^{\epsilon A}W_1,W_2e^{-\epsilon A})=L(W_1,W_2)$ for all $\epsilon$, the derivative at $0$ vanishes:
$$
0=\langle\nabla_1,AW_1\rangle-\langle\nabla_2,W_2A\rangle=\operatorname{tr}\!\big(A\,(W_1\nabla_1^\top-\nabla_2^\top W_2)\big).
$$
This holds for every $A$, so $W_1\nabla_1^\top=\nabla_2^\top W_2$, and transposing, $\nabla_1W_1^\top=W_2^\top\nabla_2$. Under $\tau\dot W_i=-\nabla_i$,
$$
\tau\tfrac{d}{dt}(W_1W_1^\top)=-(\nabla_1W_1^\top+W_1\nabla_1^\top),\qquad
\tau\tfrac{d}{dt}(W_2^\top W_2)=-(\nabla_2^\top W_2+W_2^\top\nabla_2),
$$
and the identity and its transpose show these are equal. Only the symmetry of $L$ was used.

*Which generators.* Each $\operatorname{tr}(AD)$ is conserved, but $D$ is symmetric, so $\operatorname{tr}(AD)$ sees only the symmetric part of $A$: the $h(h+1)/2$ symmetric generators (rescalings and shears) give the independent entries of $D$, and the rotations of the hidden layer, though symmetries, give nothing. In general, if $\theta\mapsto e^{\epsilon T}\theta$ and $e^{\epsilon T^\top}$ are both symmetries of $L$, then $\tau\frac{d}{dt}\tfrac12\theta^\top T\theta=-\tfrac12\theta^\top(T+T^\top)\nabla L=0$, and the charge $\tfrac12\theta^\top T\theta$ sees only the symmetric part of $T$. Here $T_A(W_1,W_2)=(AW_1,-W_2A)$, its transpose is $T_{A^\top}$ (also a symmetry), and $\tfrac12\langle\theta,T_A\theta\rangle=\tfrac12\operatorname{tr}\big(A\,(W_1W_1^\top-W_2^\top W_2)\big)=-\tfrac12\operatorname{tr}(AD)$, which vanishes for antisymmetric $A$.

**3.** With $W_1'=W_1+\eta W_2^\top E$ and $W_2'=W_2+\eta EW_1^\top$,
$$
W_1'W_1'^\top=W_1W_1^\top+\eta(W_2^\top EW_1^\top+W_1E^\top W_2)+\eta^2W_2^\top EE^\top W_2,
$$
$$
W_2'^\top W_2'=W_2^\top W_2+\eta(W_1E^\top W_2+W_2^\top EW_1^\top)+\eta^2W_1E^\top EW_1^\top .
$$
The $O(\eta)$ terms cancel in the difference, leaving $\Delta D=\eta^2(W_1E^\top EW_1^\top-W_2^\top EE^\top W_2)$. Over $O(1/\eta)$ steps with $E=O(1)$ the drift accumulates to $O(\eta)$ (the code cell below checks this).
"""
    ),
    markdown(
        r"""
<a id="td1-ex1-3"></a>

**Exercise 1.3 — Decoupling.** Assume $W_1(0)=R\operatorname{diag}(b)V^\top$ and $W_2(0)=U\operatorname{diag}(c)R^\top$ with $R$ orthogonal, and $h=n_x=n_y=r$ for simplicity.

1. Show that the right-hand sides of (1.1) have the same form, so the form is preserved with $R$, $U$ and $V$ fixed, and derive (1.3).
2. Show that (1.3) is gradient flow on $\sum_\alpha\tfrac12(s_\alpha-b_\alpha c_\alpha)^2$.
3. What changes if $h>r$? If $h<r$?

<details><summary>Hint</summary>

$U^\top U=I$ and $R^\top R=I$, so $E=U\operatorname{diag}(s-bc)V^\top$ and products telescope. For 3, think about which columns of $R$ are used.
</details>
"""
    ),
    answer(
        r"""
**1.** $W=W_2W_1=U\operatorname{diag}(c)R^\top R\operatorname{diag}(b)V^\top=U\operatorname{diag}(bc)V^\top$, so $E=U\operatorname{diag}(s-bc)V^\top$. Then
$$
W_2^\top E=R\operatorname{diag}(c)U^\top U\operatorname{diag}(s-bc)V^\top=R\operatorname{diag}\big(c(s-bc)\big)V^\top,
\qquad
EW_1^\top=U\operatorname{diag}\big(b(s-bc)\big)R^\top .
$$
The velocities have the forms $R\,(\text{diagonal})\,V^\top$ and $U\,(\text{diagonal})\,R^\top$, so the ansatz is preserved with fixed $R,U,V$, and matching diagonals gives (1.3).

**2.** $L=\tfrac12\|\Sigma-W\|_F^2=\tfrac12\|\operatorname{diag}(s-bc)\|^2=\sum_\alpha\tfrac12(s_\alpha-b_\alpha c_\alpha)^2$ by orthogonal invariance of the Frobenius norm. Then $-\partial L/\partial b_\alpha=c_\alpha(s_\alpha-b_\alpha c_\alpha)$ and $-\partial L/\partial c_\alpha=b_\alpha(s_\alpha-b_\alpha c_\alpha)$.

**3.** If $h>r$, use the first $r$ columns of $R$; hidden directions that start at zero stay at zero, because their velocity is proportional to their own weights. If $h<r$, only $h$ modes can be carried, any $h$ of them, and the others have $a_\alpha\equiv0$: these are the critical points of §3. Only the top $h$ is a local minimum, but the ansatz cannot show how the network chooses, because that requires hidden units to rotate between modes.
"""
    ),
    markdown(
        r"""
<a id="td1-ex1-4"></a>

**Exercise 1.4 — The scalar ODE and its regimes.**

1. With $a=bc$ and $d=c^2-b^2$, show that $\dot d=0$, that $b^2+c^2=\sqrt{4a^2+d^2}$, and derive (1.4).
2. Solve the balanced case $d=0$, $0<a_0<s$, and derive $t_{1/2}$. How long does the rise from $a=\epsilon s$ to $a=(1-\epsilon)s$ take, compared with the plateau?
3. For $|d|\gg s$, bound the relative change of the prefactor $\sqrt{4a^2+d^2}$ over the whole trajectory from $a_0=0$ to $a=s$. Solve the approximate equation.
4. What does a balanced mode with $a_0=0$ do? With $a_0<0$? Why does this not happen with a random initialization?

<details><summary>Hint</summary>

For 1, $(b^2+c^2)^2=(c^2-b^2)^2+4b^2c^2$. For 2, separate variables with partial fractions $\frac{1}{a(s-a)}=\frac1s\big(\frac1a+\frac1{s-a}\big)$. For 4 with $d=0$, the prefactor is $b^2+c^2=2|a|$, not $2a$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\tau\dot d=2c\,\tau\dot c-2b\,\tau\dot b=2cb(s-a)-2bc(s-a)=0$. Next, $(b^2+c^2)^2=(c^2-b^2)^2+4b^2c^2=d^2+4a^2$, and $b^2+c^2\ge0$, so $b^2+c^2=\sqrt{4a^2+d^2}$. Finally $\tau\dot a=\tau\dot b\,c+b\,\tau\dot c=(c^2+b^2)(s-a)=\sqrt{4a^2+d^2}\,(s-a)$.

**2.** $\tau\,da/(a(s-a))=2\,dt$. With the partial fractions in the hint, $\frac{\tau}{s}\ln\frac{a}{s-a}\Big|_{a_0}^{a}=2t$, so
$$
\frac{a}{s-a}=\frac{a_0}{s-a_0}\,e^{2st/\tau}\quad\Longrightarrow\quad a(t)=\frac{s\,e^{2st/\tau}}{e^{2st/\tau}-1+s/a_0}.
$$
Setting $a=s/2$ gives $e^{2st_{1/2}/\tau}=s/a_0-1$, so $t_{1/2}=\frac{\tau}{2s}\ln(s/a_0-1)$. The rise from $\epsilon s$ to $(1-\epsilon)s$ takes $\frac{\tau}{2s}\big[\ln\frac{1-\epsilon}{\epsilon}-\ln\frac{\epsilon}{1-\epsilon}\big]=\frac{\tau}{s}\ln\frac{1-\epsilon}{\epsilon}$, independent of $a_0$. As $a_0\to0$ the plateau grows like $\ln(1/a_0)$ while the rise stays at about $\tau/s$, so the curve looks like a step. This is Saxe et al.'s Eq. 11–12, with their $u$ for our $a$.

**3.** On $a\in[0,s]$, $|d|\le\sqrt{4a^2+d^2}\le\sqrt{4s^2+d^2}=|d|\big(1+2s^2/d^2+O(s^4/d^4)\big)$. So the prefactor changes by a relative amount at most about $2s^2/d^2$. Replacing it by $|d|$ gives $\tau\dot a=|d|(s-a)$, so
$$
a(t)=s-(s-a_0)\,e^{-|d|t/\tau},
$$
an exponential with a rate independent of $s$, with an $O(s^2/d^2)$ error in the rate.

**4.** With $d=0$ and $a_0=0$ we have $b=c=0$, a fixed point (the saddle of §3), and the mode never learns. With $d=0$ and $a_0<0$, $b=-c$ and the prefactor is $2|a|$, so $\tau\dot a=2|a|(s-a)>0$: $a$ increases toward $0$ but never crosses it, because the prefactor vanishes there. The mode is stuck on the wrong side of the saddle. A random initialization has input and output weights that are not exactly parallel, so its invariant $\Delta$ (§4.3) is positive and the prefactor $\sqrt{4a^2+\Delta^2}$ never vanishes: exact balance with $a_0\le0$ is a measure-zero trap.
"""
    ),
    markdown(
        r"""
<a id="td1-ex1-5"></a>

**Exercise 1.5 (code) — Integrate the mode ODEs and compare with the closed forms.**
Integrate the **unreduced** system (1.3) for one mode, $(b,c)$, from $bc=a_0=10^{-3}$ with
$d=0$ and with $d=20$, for $s\in\{3,1.5,0.75\}$. Compare $a=bc$ with the logistic
solution ($d=0$) and with the closed form $t(a)$ ($d=20$), and check that $d$ stays
constant. The helpers `rk4`, `logistic` and `t_of_a` are defined in the setup cell.
"""
    ),
    code_answer(
        r"""
def mode_rhs(y, s):
    b, c = y
    # TODO: return the right-hand side of (1.3) for one mode, as an array [db/dt, dc/dt] (tau = 1)
    raise NotImplementedError

t = np.linspace(0, 8, 401)
# TODO: for d in (0, 20) and s in (3, 1.5, 0.75): choose b0, c0 with b0*c0 = 1e-3 and c0^2 - b0^2 = d,
#       integrate with rk4, compare a = b*c with logistic(...) or t_of_a(...), and plot a(t).
""",
        r"""
def mode_rhs(y, s):
    b, c = y
    return np.array([c * (s - b * c), b * (s - b * c)])

t = np.linspace(0, 8, 401)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for s, col in zip([3.0, 1.5, 0.75], ["C0", "C1", "C2"]):
    for d, ax in zip([0.0, 20.0], axs):
        a0 = 1e-3
        c0 = np.sqrt((d + np.sqrt(d**2 + 4 * a0**2)) / 2); b0 = a0 / c0   # b0 c0 = a0, c0^2 - b0^2 = d
        Y = rk4(lambda y: mode_rhs(y, s), [b0, c0], t)
        a = Y[:, 0] * Y[:, 1]
        ax.plot(t, a, color=col, lw=2, label=f"s = {s}")
        if d == 0:
            ax.plot(t, logistic(t, s, a0), "k:", lw=1)
            ax.plot(np.log(s / a0 - 1) / (2 * s), s / 2, "o", color=col)   # predicted half-time
            err = np.max(np.abs(a - logistic(t, s, a0)))
            print(f"d = 0,  s = {s}: max |a - logistic|          = {err:.1e}")
        else:
            m = a < s * (1 - 1e-9)                                           # avoid log(0) at a = s
            err = np.max(np.abs(t_of_a(a[m], s, d, a0) - t[m]))
            print(f"d = 20, s = {s}: max |t_closed(a) - t| / tau = {err:.1e}")
        print(f"              drift of d over the run: {np.max(np.abs(Y[:, 1]**2 - Y[:, 0]**2 - d)):.1e}")
axs[0].set_title("d = 0: sigmoids ordered by s (dotted: logistic)")
axs[1].set_title("d = 20: exponentials at a common rate")
axs[1].set_xlim(0, 0.6)
for ax in axs:
    ax.set_xlabel("t / tau"); ax.set_ylabel("a"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""",
    ),
    markdown(
        r"""
The next cell checks Exercise 1.2.3: gradient descent conserves $D$ only up to a drift
that, at a fixed training time $t/\tau$, shrinks in proportion to $\eta$.
"""
    ),
    code(
        r"""
Sigma, U, V = make_target([3.0, 1.5, 0.75])
r = np.random.default_rng(1)
W1i, W2i = 0.3 * r.normal(size=(6, 6)), 0.3 * r.normal(size=(6, 6))
Dmat = lambda W1, W2: W2.T @ W2 - W1 @ W1.T
for eta in [0.04, 0.02, 0.01, 0.005]:
    steps = int(round(4 / eta))                       # train to t = 4 tau
    _, hist = gd([W1i, W2i], Sigma, eta, steps, every=steps)
    drift = np.linalg.norm(Dmat(*hist[-1]) - Dmat(W1i, W2i))
    print(f"eta = {eta:<6} ||D(4 tau) - D(0)||_F = {drift:.2e}   drift / eta = {drift / eta:.3f}")
"""
    ),
    markdown(
        r"""
<a id="td1-ex1-6"></a>

**Exercise 1.6 — Deriving alignment from small random initialization.**

1. Linearize (1.1) for small $W$ and show $\tau\dot Z=ZM$.
2. Find the eigenvalues and eigenvectors of $M$.
3. With $e^\pm_\alpha:=(v_\alpha,\pm u_\alpha)/\sqrt2$ and $z^\pm_\alpha:=Z(0)e^\pm_\alpha$, compute $p_\alpha(t)=W_1(t)v_\alpha$, $q_\alpha(t)=W_2(t)^\top u_\alpha$ and $u_\alpha^\top W(t)v_\beta$.
4. Show that $p_\alpha$ and $q_\alpha$ become parallel and of equal length. Estimate the ratio $a_2/a_1$ at the time when $a_1$ reaches $s_1/2$, as a function of the initial scale.
5. *(Optional, uses §4–§5.)* Match onto §1.4: what is the effective $a_0$ of the first mode to be learned, and how does it compare with $\Delta/4$?

<details><summary>Hint</summary>

For 3, write $(v_\beta,0)=(e_\beta^++e_\beta^-)/\sqrt2$ and $(0,u_\alpha)=(e_\alpha^+-e_\alpha^-)/\sqrt2$, and use $Z(t)e^\pm_\alpha=e^{\pm s_\alpha t/\tau}z^\pm_\alpha$. For 4, use the balanced half-time with the effective $a_0$.
</details>
"""
    ),
    answer(
        r"""
**1.** Dropping the cubic terms ($E\approx\Sigma$), $\tau\dot W_1=W_2^\top\Sigma$ and $\tau\dot W_2^\top=W_1\Sigma^\top$. With $Z=[Z_1,Z_2]=[W_1,W_2^\top]$, $ZM=[Z_2\Sigma,\;Z_1\Sigma^\top]=[W_2^\top\Sigma,\;W_1\Sigma^\top]=\tau\dot Z$.

**2.** $M(v_\alpha,\pm u_\alpha)=(\Sigma^\top(\pm u_\alpha),\,\Sigma v_\alpha)=(\pm s_\alpha v_\alpha,\,s_\alpha u_\alpha)=\pm s_\alpha(v_\alpha,\pm u_\alpha)$. So $\pm s_\alpha$ with eigenvectors $e^\pm_\alpha=(v_\alpha,\pm u_\alpha)/\sqrt2$, and eigenvalue $0$ on the null directions of $\Sigma$ and $\Sigma^\top$ (if $n_x+n_y>2r$).

**3.** $Z(t)=Z(0)e^{Mt/\tau}$, so $Z(t)e^\pm_\alpha=e^{\pm s_\alpha t/\tau}z^\pm_\alpha$. Then
$$
p_\beta=Z(t)(v_\beta,0)=\tfrac1{\sqrt2}\big(e^{s_\beta t/\tau}z^+_\beta+e^{-s_\beta t/\tau}z^-_\beta\big),\qquad
q_\alpha=Z(t)(0,u_\alpha)=\tfrac1{\sqrt2}\big(e^{s_\alpha t/\tau}z^+_\alpha-e^{-s_\alpha t/\tau}z^-_\alpha\big),
$$
and $u_\alpha^\top Wv_\beta=q_\alpha\cdot p_\beta$:
$$
u_\alpha^\top W(t)v_\beta=\tfrac12\Big[(z^+_\alpha\cdot z^+_\beta)e^{(s_\alpha+s_\beta)t/\tau}+(z^+_\alpha\cdot z^-_\beta)e^{(s_\alpha-s_\beta)t/\tau}-(z^-_\alpha\cdot z^+_\beta)e^{(s_\beta-s_\alpha)t/\tau}-(z^-_\alpha\cdot z^-_\beta)e^{-(s_\alpha+s_\beta)t/\tau}\Big].
$$
For $\alpha=\beta$ the middle terms cancel, leaving $a_\alpha=\tfrac12\big(|z^+_\alpha|^2e^{2s_\alpha t/\tau}-|z^-_\alpha|^2e^{-2s_\alpha t/\tau}\big)$. After a few multiples of $\tau/s$ the leading term dominates: $u_\alpha^\top Wv_\beta\approx\tfrac12(z^+_\alpha\cdot z^+_\beta)e^{(s_\alpha+s_\beta)t/\tau}$.

**4.** The decaying parts die out, so $p_\alpha$ and $q_\alpha$ both approach $e^{s_\alpha t/\tau}z^+_\alpha/\sqrt2$: same direction, same length. The misalignment and imbalance do not disappear (they are encoded in the conserved invariant $\Delta_\alpha$ of §4.3), but they become negligible relative to the mode's kernel $k_\alpha=|p_\alpha|^2+|q_\alpha|^2\approx|z^+_\alpha|^2e^{2s_\alpha t/\tau}$. With $a_{0,\alpha}:=\tfrac12|z^+_\alpha|^2$, mode 1 reaches $s_1/2$ at $t_1\approx(\tau/2s_1)\ln(s_1/a_{0,1})$, i.e. $e^{2s_1t_1/\tau}\approx s_1/a_{0,1}$. Mode 2 is still in its exponential phase, $a_2\approx a_{0,2}e^{2s_2t_1/\tau}$, so
$$
\frac{a_2}{a_1}\approx\frac{2a_{0,2}}{s_1}\Big(\frac{s_1}{a_{0,1}}\Big)^{s_2/s_1}=2\,\frac{a_{0,2}}{a_{0,1}}\Big(\frac{a_{0,1}}{s_1}\Big)^{1-s_2/s_1}.
$$
Relative to $a_1$, the off-diagonal entries $u_1^\top Wv_2$, $u_2^\top Wv_1$ are suppressed by the square root of that factor (rate $s_1+s_2$ versus $2s_1$), times the overlap $z^+_1\cdot z^+_2/|z^+_1|^2=O(h^{-1/2})$; all of this vanishes as a power of the initial scale. Relative to $a_2$, though, the cross term grows like $e^{(s_1-s_2)t/\tau}$: it is small only in absolute size (about $h^{-1/2}\sqrt{a_1a_2}$), and its removal after mode 1 saturates is nonlinear, so the notebook checks that step numerically.

**5.** Once the decaying parts are gone, the first mode is an aligned balanced mode with strength $\tfrac12|z^+_\alpha|^2e^{2s_\alpha t/\tau}$, the small-$a$ behaviour of the logistic with $a_0=\tfrac12|z^+_\alpha|^2$. Exactly, $z^+_\alpha=(p_\alpha(0)+q_\alpha(0))/\sqrt2$, so $\tfrac12|z^+_\alpha|^2=\tfrac14\big(k_\alpha(0)+2a_\alpha(0)\big)$. For Gaussian weights at large width, $|a_\alpha(0)|=O(\sigma^2\sqrt h)\ll k_\alpha(0)\approx2\sigma^2h$, so the effective $a_0$ is $\approx\sigma^2h/2\approx\Delta_\alpha/4$. (This needs large width; shrinking $\sigma$ at fixed $h$ does not make $a(0)/k(0)$ small.) That is the effective initial strength the exact solution of (4.3) gives as $a_0\to0$ (Exercise 5.2), so the two analyses agree.
"""
    ),
    markdown(
        r"""
**Code: watching alignment happen before the loss moves.** Small Gaussian initialization
($h=8$, $\sigma=10^{-3}$). For each mode, the cell plots the cosine between
$p_\alpha=W_1v_\alpha$ and $q_\alpha=W_2^\top u_\alpha$, and the fraction of $\|U^\top WV\|_F$
that is off-diagonal. The loss is in a separate panel.
"""
    ),
    code(
        r"""
svals = [3.0, 2.0, 1.0, 0.5]
Sigma, U, V = make_target(svals)
r = np.random.default_rng(4); h, sig = 8, 1e-3
ts, hist = gd([sig * r.normal(size=(h, 6)), sig * r.normal(size=(6, h))], Sigma, 0.01, 800, every=5)
cosang, offd, loss = [], [], []
for W1t, W2t in hist:
    P, Q = W1t @ V[:, :4], W2t.T @ U[:, :4]
    cosang.append(np.sum(P * Q, 0) / (np.linalg.norm(P, axis=0) * np.linalg.norm(Q, axis=0)))
    Mt = U.T @ (W2t @ W1t) @ V
    offd.append(np.linalg.norm(Mt - np.diag(np.diag(Mt))) / np.linalg.norm(Mt))
    loss.append(0.5 * np.sum((Sigma - W2t @ W1t) ** 2))
cosang = np.array(cosang)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
for i in range(4):
    axs[0].plot(ts, cosang[:, i], label=f"mode {i + 1} (s = {svals[i]})")
axs[0].set_title("cos(p, q): each mode's in/out weights align"); axs[0].legend(fontsize=7)
axs[1].semilogy(ts, offd); axs[1].set_title("off-diagonal fraction of U^T W V")
axs[2].plot(ts, loss); axs[2].set_title("loss")
for ax in axs: ax.set_xlabel("t / tau")
plt.tight_layout(); plt.show()
k1 = np.searchsorted(ts, 1.0)
print(f"at t = tau: cos(p,q) = {np.round(cosang[k1], 3)}, off-diagonal fraction = {offd[k1]:.3f}, "
      f"loss = {loss[k1]:.3f} (initial {loss[0]:.3f}, final {loss[-1]:.3f})")
"""
    ),
    # ================================================================== §2
    markdown(
        r"""
<a id="td1-s2"></a>

## 2. Depth

Extra linear layers add nothing to what the network can represent. What do they change?
With $N$ layers, $W=W_N\cdots W_1$, a balanced mode of strength $a$ and target singular value
$s$ obeys $\tau\dot a=Na^{2-2/N}(s-a)$. From depth 3, the plateau grows as a power of $1/a_0$
($a_0$ the initial strength) instead of logarithmically, and the largest stable learning
rate falls like $1/N$. A side branch: nothing later depends on it.

**Papers:** the depth-$N$ equation is Saxe et al.'s (2014, Eq. 15 and App. B); Jacot et al.
(2021) on saddle-to-saddle escape (see [Readings](#td1-readings)).
"""
    ),
    markdown(
        r"""
<a id="td1-s2-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* as in §1: whitened inputs, $\Sigma=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$, error
$E=\Sigma-W$, mode strength $a=u^\top Wv$, $\tau=1/\eta$. At depth 2 a balanced mode has a
plateau of about $(\tau/2s)\ln(s/a_0)$.

**2.1 The depth-$N$ mode equation.** With aligned layers $W_j=R_j\operatorname{diag}(w_j)R_{j-1}^\top$
($R_j$ orthogonal, $R_0=V$, $R_N=U$), mode $\alpha$ is carried by one scalar per layer, with
$a=w_1\cdots w_N$, $\tau\dot w_j=\big(\prod_{i\ne j}w_i\big)(s-a)$ and conserved differences
$w_{j+1}^2-w_j^2$ (Exercise 2.1). Balanced, on the positive branch ($a>0$),

$$
\boxed{\;\tau\dot a=N\,a^{2-2/N}\,(s-a)\;} \tag{2.1}
$$

(Saxe et al.'s Eq. 15 with $N_l-1=N$). The prefactor is the mode kernel
$k=\sum_j\prod_{i\ne j}w_i^2$. With imbalance, $k$ is dominated by the products that omit a
small layer: if all layers but one are large the mode is lazy; if only one is large, the
others form a rich sub-network. (The mode explorer's slider sets every $w_{j+1}^2-w_j^2$ to $d$.)

**2.2 Plateaus grow as a power of $1/a_0$.** For $a\ll s$, $\tau\dot a\approx Ns\,a^{2-2/N}$, which
for $N\ge3$ blows up in finite time (Exercise 2.2):

$$
t_{\rm esc}\approx\frac{\tau}{(N-2)\,s}\;a_0^{-(1-2/N)}\qquad(N\ge3).
$$

With a per-layer scale $w_0$ ($a_0=w_0^N$) this is $w_0^{-(N-2)}\tau/((N-2)s)$, exponential in
depth. Even at fixed end-to-end $a_0$ the plateau grows with depth up to
$N\approx2\ln(1/a_0)-2$. So the end-to-end map should not shrink with depth, and should start
comparable to the target to avoid a long plateau.

**2.3 Learning rate and depth.** At the balanced minimum the Hessian of
$\tfrac12(s-\prod_jw_j)^2$ is $gg^\top$ with $g_j=\partial a/\partial w_j=s^{(N-1)/N}$, whose nonzero
eigenvalue is $|g|^2=Ns^{2-2/N}$. Stability needs $\eta|g|^2<2$, so the largest stable learning
rate scales as $1/N$; at a fixed fraction of it, the final contraction per step does not
depend on $N$. Saxe et al. (App. B) combine this with an initialization whose end-to-end
strength does not shrink with depth, and find learning times bounded as $N\to\infty$ (the
remaining delay grows like $s/a_0$, so it is short only if $a_0$ is not small). Their choice is
orthogonal layers; §6 (item 4) discusses what that guarantees.

<a id="td1-ex2-1"></a>

**Exercise 2.1 — Derive the depth-$N$ equation.**

1. Show that $\nabla_{W_j}L=-(W_N\cdots W_{j+1})^\top E\,(W_{j-1}\cdots W_1)^\top$.
2. With the aligned ansatz, derive the per-layer scalar equations and show that $w_{j+1}^2-w_j^2$ is conserved.
3. Derive (2.1) in the balanced case, and show that in general $\tau\dot a=k\,(s-a)$ with $k=\sum_j\prod_{i\ne j}w_i^2$.

<details><summary>Hint</summary>

For 1, perturb one factor in the product, as in Exercise 1.1. For 2, the orthogonal $R_j$ cancel in pairs inside every product. Compute $\tau\frac{d}{dt}w_j^2$.
</details>
"""
    ),
    answer(
        r"""
**1.** Write $W=A\,W_j\,B$ with $A=W_N\cdots W_{j+1}$ and $B=W_{j-1}\cdots W_1$ (empty products are identities). Perturbing $W_j$ by $\delta$ changes $W$ by $A\delta B$, so $\delta L=\langle -E,A\delta B\rangle=\langle -A^\top EB^\top,\delta\rangle$.

**2.** With $W_j=R_j\operatorname{diag}(w_j)R_{j-1}^\top$ the products telescope: $A=U\operatorname{diag}(\prod_{i>j}w_i)R_j^\top$, $B=R_{j-1}\operatorname{diag}(\prod_{i<j}w_i)V^\top$, and $E=U\operatorname{diag}(s-a)V^\top$. So
$$
-\nabla_{W_j}L=A^\top EB^\top=R_j\operatorname{diag}\Big(\prod_{i\ne j}w_i\,(s-a)\Big)R_{j-1}^\top,
$$
which has the ansatz form, so $\tau\dot w_j=\prod_{i\ne j}w_i\,(s-a)$ for each mode. Then $\tau\tfrac{d}{dt}w_j^2=2\prod_iw_i\,(s-a)=2a(s-a)$ for every $j$, so all the $w_j^2$ move together and their differences are conserved.

**3.** $\tau\dot a=\sum_j\big(\prod_{i\ne j}w_i\big)\tau\dot w_j=\sum_j\big(\prod_{i\ne j}w_i\big)^2(s-a)=k\,(s-a)$. Balanced: $w_i=a^{1/N}$ for all $i$, so each term is $a^{2(N-1)/N}$ and $k=Na^{2-2/N}$.
"""
    ),
    markdown(
        r"""
<a id="td1-ex2-2"></a>

**Exercise 2.2 — Plateau scaling.**

1. Solve $\tau\dot a=Ns\,a^{2-2/N}$ (the small-$a$ limit of (2.1)) for $N\ge3$ and derive $t_{\rm esc}$.
2. Compare $N=2$ and $N=3$ at $a_0=10^{-6}$, $s=1$. Then rewrite both in terms of the per-layer scale $w_0$ and compare $N=3$ with $N=6$ at $w_0=0.1$.
3. *(Optional.)* Why is the blow-up time in 1 a good estimate of the time to reach $a\approx s/2$ when $a_0\ll s$?

<details><summary>Hint</summary>

Let $p=1-2/N$, so $a^{2-2/N}=a^{1+p}$, and $\frac{d}{dt}a^{-p}=-p\,a^{-p-1}\dot a$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $p=1-2/N\in(0,1)$, $\tau\frac{d}{dt}a^{-p}=-p\,a^{-p-1}\cdot Ns\,a^{1+p}=-pNs$. So $a(t)^{-p}=a_0^{-p}-pNs\,t/\tau$, which reaches zero ($a\to\infty$ in the approximation) at
$$
t_{\rm esc}=\frac{\tau\,a_0^{-p}}{pNs}=\frac{\tau}{(N-2)s}\,a_0^{-(1-2/N)}.
$$

**2.** At $a_0=10^{-6}$, $s=1$: $N=2$ gives $\tfrac12\ln10^6\approx6.9\,\tau$; $N=3$ gives $(10^{-6})^{-1/3}=100\,\tau$. In per-layer terms: $N=2$, $\tfrac{\tau}{2s}\ln(s/w_0^2)=\tfrac{\tau}{s}\ln(1/w_0)$ (for $s=1$); $N\ge3$, $\tau\,w_0^{-(N-2)}/((N-2)s)$. At $w_0=0.1$: $N=3$ gives $10\,\tau$, and $N=6$ gives $10^4/4=2500\,\tau$.

**3.** The approximation drops the factor $(s-a)/s\approx1$ for $a\ll s$, where almost all the time is spent. The final stretch from $s/10$ to $s/2$ takes only $O(\tau/(Ns^{2-2/N}))$, independent of $a_0$, so $t_{1/2}=t_{\rm esc}\,(1+o(1))$ as $a_0\to0$.
"""
    ),
    markdown(
        r"""
<a id="td1-ex2-3"></a>

**Exercise 2.3 (code) — Plateau length versus initialization and depth.** For $N=2,3,4$
and $a_0\in[10^{-6},10^{-1}]$, integrate (2.1) with $s=1$ and record the half-time
$t_{1/2}$. Plot $t_{1/2}$ against $a_0$ on log–log axes together with the predictions
$\tfrac12\ln(1/a_0)$ and $a_0^{-(1-2/N)}/(N-2)$.
"""
    ),
    code_answer(
        r"""
def half_time_depth(N, s, a0):
    # TODO: integrate tau a' = N a^(2-2/N) (s - a) from a0 and return the time at which a = s/2.
    # Tip: integrate in log-time or with an adaptive step; plateaus can be long.
    raise NotImplementedError

# TODO: loop over N in (2, 3, 4) and a log-spaced grid of a0; plot t_1/2 versus a0 with the predictions.
""",
        r"""
def half_time_depth(N, s, a0):
    # Integrate in the variable x = ln a, where dx/dt = N a^(1-2/N) (s - a); step in x, accumulate t.
    xs = np.linspace(np.log(a0), np.log(s / 2), 4001)
    a = np.exp(xs)
    dtdx = 1.0 / (N * a**(1 - 2 / N) * (s - a))      # dt/dx, smooth in x
    return np.sum((dtdx[1:] + dtdx[:-1]) / 2 * np.diff(xs))   # trapezoid rule

s, a0s = 1.0, np.logspace(-6, -1, 21)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for N, col in zip([2, 3, 4], ["C0", "C1", "C2"]):
    th = np.array([half_time_depth(N, s, a0) for a0 in a0s])
    pred = 0.5 * np.log(s / a0s - 1) / s if N == 2 else a0s**(-(1 - 2 / N)) / ((N - 2) * s)
    ax.loglog(a0s, th, "o", color=col, ms=4, label=f"N = {N}, numerical")
    ax.loglog(a0s, pred, "-", color=col, lw=1.5, label=f"N = {N}, prediction")
    print(f"N = {N}: t_1/2 at a0 = 1e-6 is {th[0]:9.1f} tau (prediction {pred[0]:9.1f})")
ax.set_xlabel("initial mode strength a0"); ax.set_ylabel("half-time t_1/2 / tau")
ax.set_title("plateau length: logarithmic for N = 2, a power law for N >= 3")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.show()
""",
    ),
    markdown(
        r"""
The power-law prediction is the leading term as $a_0\to0$. At $a_0\sim10^{-1}$ the
blow-up time overshoots the time to reach $s/2$, so the points sit below the line there.
"""
    ),
    # ================================================================== §3
    markdown(
        r"""
<a id="td1-s3"></a>

## 3. The landscape

What does "no bad local minima" buy you? For deep linear networks with square loss, every
critical point is a partial SVD reconstruction $\Sigma_S=\sum_{\alpha\in S}s_\alpha u_\alpha v_\alpha^\top$
of the target correlation $\Sigma=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$, every local minimum is
global, and the other critical points are saddles (some degenerate from depth 3). None of
this says how long training takes. A side branch: nothing later depends on it.

**Papers:** Achour, Malgouyres & Gerchinovitz (2024) on the critical points; Arora, Cohen,
Golowich & Hu (2019) on convergence (see [Readings](#td1-readings)).
"""
    ),
    markdown(
        r"""
<a id="td1-s3-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* depth-$N$ network $W=W_N\cdots W_1$, whitened inputs,
$L=\tfrac12\|\Sigma-W\|_F^2+\text{const}$, $E=\Sigma-W$, distinct singular values $s_\alpha$.
Gradient: $\nabla_{W_j}L=-(W_N\cdots W_{j+1})^\top E(W_{j-1}\cdots W_1)^\top$ (Exercise 2.1). $r_{\max}$ is
the narrowest width, so $\operatorname{rank}W\le r_{\max}$. Without whitening,
$\Sigma_{xx}=\mathbb E[xx^\top]$, $\Sigma_{yx}=\mathbb E[yx^\top]$, $\Sigma_{xy}=\Sigma_{yx}^\top$. $\tau=1/\eta$.

**3.1 Critical points are partial reconstructions.** At a critical point $W^\top E=0$ and
$EW^\top=0$, so $\Sigma=W+E$ splits into two non-overlapping pieces, and (Exercise 3.1)

$$
W=\Sigma_S:=\sum_{\alpha\in S}s_\alpha u_\alpha v_\alpha^\top,\qquad L=\tfrac12\sum_{\alpha\notin S}s_\alpha^2+\text{const},
$$

for some subset $S$ with $|S|\le r_{\max}$, at any depth. Without whitening (Achour et al.) the
projection is onto eigenvectors of $\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}$.

**3.2 Minima and the gauge manifold.** By Eckart–Young (the best rank-$k$ approximation keeps
the top $k$ singular triples), the global minima are $S=\{1,\dots,m\}$ with
$m:=\min(r,r_{\max})$; every other critical point is a saddle (Baldi–Hornik, Kawaguchi). The
minima are not isolated: $(G_1W_1,\;G_2W_2G_1^{-1},\dots,W_NG_{N-1}^{-1})$ is a minimum for any
invertible $G_j$. For two square layers and invertible $\Sigma$ the minima form one $GL(n)$
orbit of dimension $n^2$ (Exercise 3.2). This gauge symmetry makes every minimum degenerate,
which singular learning theory measures with the learning coefficient (RLCT), known exactly
for deep linear networks (Aoyagi, as cited in
[Pepin Lehalleur and Rimányi](https://arxiv.org/abs/2411.19920)); see the MDL day's MDL3,
[§2 *Occam, two ways*](../notebooks/mdl3_program_synthesis.html#wb4-section-2).

**3.3 Strict and degenerate saddles.** At the origin of one mode, depth 2 gives the loss
$\tfrac12(s-bc)^2$, with Hessian eigenvalues $\pm s$: a *strict* saddle, escaped exponentially
(§1). Depth 3 gives $\tfrac12(s-w_1w_2w_3)^2$, whose Hessian vanishes there: a *degenerate*
saddle, with the power-law escape of §2.2 (Exercise 3.3). In general (Achour et al., Theorem
7, for full-rank $\Sigma_{yx}$ with distinct singular values), a critical point that skips one
of the top modes is a strict saddle at any depth; one that keeps exactly the top $|S|<m$
modes is strict at depth 2 and can be either from depth 3 on, including $W=0$.

**3.4 Why this says little about time.** "Every local minimum is global" rules out one kind
of trap. It is not a convergence theorem (degenerate saddles and the behaviour at infinity
need separate arguments), and it says nothing about time, which is dominated by passing
near saddles: about $(\tau/2s)\ln(s/a_0)$ at depth 2 and $\propto a_0^{-(1-2/N)}$ at depth
$N\ge3$, unbounded as $a_0\to0$. Strict-saddle escape theorems fail from depth 3, and exactly
symmetric starts can sit on a saddle forever (Exercise 1.4.4). The landscape says where
training goes, not how long it stays.

<a id="td1-ex3-1"></a>

**Exercise 3.1 — Critical points are partial SVD reconstructions.**

1. From the gradient formula, show that at a critical point of any depth, $W^\top E=0$ and $EW^\top=0$.
2. Deduce that $W=\Sigma_S$ for some subset $S$ of modes, assuming the $s_\alpha$ are distinct.
3. Compute the loss at each critical point. Which are global minima?

<details><summary>Hint</summary>

For 1, multiply the $j=N$ condition on the right by $W_N^\top$ and the $j=1$ condition on the left by $W_1^\top$. For 2, $W^\top E=0$ means the column space of $E$ is orthogonal to that of $W$; similarly for rows. If $W=\sum_i\sigma_ip_iq_i^\top$ and $E=\sum_k\rho_k\tilde p_k\tilde q_k^\top$ are SVDs, what is $\Sigma=W+E$ as a sum of rank-one terms?
</details>
"""
    ),
    answer(
        r"""
**1.** The $j=N$ condition is $E\,(W_{N-1}\cdots W_1)^\top=0$. Multiplying on the right by $W_N^\top$ gives $EW^\top=0$. The $j=1$ condition is $(W_N\cdots W_2)^\top E=0$. Multiplying on the left by $W_1^\top$ gives $W^\top E=0$.

**2.** Take SVDs $W=\sum_i\sigma_ip_iq_i^\top$ and $E=\sum_k\rho_k\tilde p_k\tilde q_k^\top$ with positive $\sigma_i$, $\rho_k$. $W^\top E=0$ gives $\sum_{i,k}\sigma_i\rho_k(p_i^\top\tilde p_k)\,q_i\tilde q_k^\top=0$. The matrices $q_i\tilde q_k^\top$ are linearly independent (each family is orthonormal), so $p_i^\top\tilde p_k=0$ for all $i,k$. Likewise $EW^\top=0$ gives $q_i^\top\tilde q_k=0$. So $\{p_i\}\cup\{\tilde p_k\}$ and $\{q_i\}\cup\{\tilde q_k\}$ are orthonormal sets, and
$$
\Sigma=W+E=\sum_i\sigma_ip_iq_i^\top+\sum_k\rho_k\tilde p_k\tilde q_k^\top
$$
is an SVD of $\Sigma$. Because the singular values of $\Sigma$ are distinct, its SVD is unique up to signs, so the $(\sigma_i,p_i,q_i)$ are a subset of the $(s_\alpha,u_\alpha,v_\alpha)$. Hence $W=\Sigma_S$ with $|S|=\operatorname{rank}W\le r_{\max}$.

**3.** $E=\Sigma-\Sigma_S=\Sigma_{S^c}$, so $L=\tfrac12\sum_{\alpha\notin S}s_\alpha^2+\text{const}$. This is smallest when $S$ is the top $\min(r,r_{\max})$ modes: those are the global minima. Exercise 3.3 shows the rest are saddles, and the cited theorems show there are no other local minima.
"""
    ),
    markdown(
        r"""
<a id="td1-ex3-2"></a>

**Exercise 3.2 — The manifold of minima.** Take $N=2$ and $n_x=n_y=h=n$ with $\Sigma$ invertible.

1. Show that the set of global minima is exactly $\{(GW_1^*,\,W_2^*G^{-1}):G\in GL(n)\}$ for any one minimizer $(W_1^*,W_2^*)$. What is its dimension?
2. Show that at a global minimum the Hessian of $L$ has exactly $n^2$ zero eigenvalues and $n^2$ positive ones.
3. *(Optional.)* Which minima are balanced? Show that the balanced minima form an orbit of the orthogonal group $O(n)$.

<details><summary>Hint</summary>

For 1, $W_2W_1=\Sigma$ with $\Sigma$ invertible forces both factors to be invertible; set $G=W_1W_1^{*-1}$. For 2, at a minimum $E=0$, so the Hessian is the Gauss–Newton term $\delta\mapsto\|\delta W_2\,W_1+W_2\,\delta W_1\|_F^2$. Count the dimension of its kernel.
</details>
"""
    ),
    answer(
        r"""
**1.** If $W_2W_1=\Sigma$ is invertible, then $W_1$ and $W_2$ are invertible $n\times n$ matrices. Given another minimizer, set $G=W_1W_1^{*-1}$. Then $W_1=GW_1^*$ and $W_2=\Sigma W_1^{-1}=W_2^*W_1^*W_1^{-1}=W_2^*G^{-1}$. Conversely every such pair has product $\Sigma$. The map $G\mapsto(GW_1^*,W_2^*G^{-1})$ is injective, so the set is a copy of $GL(n)$, of dimension $n^2$.

**2.** Around a minimum, $(W_2+\delta_2)(W_1+\delta_1)=\Sigma+\delta_2W_1+W_2\delta_1+\delta_2\delta_1$, so $L=\tfrac12\|\delta_2W_1+W_2\delta_1\|^2+O(\delta^3)$. The Hessian is the quadratic form $Q(\delta)=\|\delta_2W_1+W_2\delta_1\|^2\ge0$, with kernel $\{\delta_2W_1+W_2\delta_1=0\}$. For any $\delta_1$, $\delta_2=-W_2\delta_1W_1^{-1}$ solves this, so the kernel has dimension $n^2$, matching the tangent space of the orbit ($\delta_1=AW_1$, $\delta_2=-W_2A$). The map $(\delta_1,\delta_2)\mapsto\delta_2W_1+W_2\delta_1$ is onto (take $\delta_1=0$, $\delta_2=MW_1^{-1}$), so the other $n^2$ eigenvalues are positive. So a flatness measure that does not quotient out the symmetry measures the parametrization, not the function.

**3.** $D=W_2^\top W_2-W_1W_1^\top=0$. From a balanced minimizer $(W_1^*,W_2^*)$, the minimizer $(GW_1^*,W_2^*G^{-1})$ has $D=G^{-\top}W_2^{*\top}W_2^*G^{-1}-GW_1^*W_1^{*\top}G^\top$. If $G$ is orthogonal this is $G(W_2^{*\top}W_2^*-W_1^*W_1^{*\top})G^\top=0$. Conversely, for a balanced minimizer $\Sigma^\top\Sigma=W_1^\top W_2^\top W_2W_1=W_1^\top W_1W_1^\top W_1=(W_1^\top W_1)^2$, so $W_1^\top W_1=(\Sigma^\top\Sigma)^{1/2}$ and $W_1=O\,(\Sigma^\top\Sigma)^{1/4}$ with $O$ orthogonal (polar decomposition); all of them are related by $G\in O(n)$. The balanced minima form an $n(n-1)/2$-dimensional orbit, and since $D$ is conserved, gradient flow from a balanced start can only end there.
"""
    ),
    markdown(
        r"""
<a id="td1-ex3-3"></a>

**Exercise 3.3 — Strict versus degenerate saddles.**

1. For $N=2$, compute the Hessian of $L$ at the origin $W_1=W_2=0$ in full matrix form. Show that its eigenvalues are $\pm s_\alpha$, each with multiplicity $h$, plus zeros.
2. For $N=2$, take an aligned critical point of §1 (each kept mode on its own hidden direction, unused hidden directions zero) with $W=\Sigma_S$, some $\beta\notin S$, and a spare hidden direction. Exhibit a direction of negative curvature.
3. For $N=3$, show that the Hessian at the origin is zero.

<details><summary>Hint</summary>

For 1, $L=\tfrac12\|\Sigma\|^2-\langle\Sigma,W_2W_1\rangle+\tfrac12\|W_2W_1\|^2$, and the last term is quartic. For 2, use a free hidden direction $e$ and perturb $W_1\to W_1+\epsilon\,e\,v_\beta^\top$ and $W_2\to W_2+\epsilon\,u_\beta e^\top$.
</details>
"""
    ),
    answer(
        r"""
**1.** Near the origin $L=\text{const}-\langle\Sigma,W_2W_1\rangle+O(\|W\|^4)$, so the Hessian is the quadratic form of $-\langle\Sigma,W_2W_1\rangle$. Give hidden unit $k$ input weights $p_k\in\mathbb R^{n_x}$ (row $k$ of $W_1$) and output weights $q_k\in\mathbb R^{n_y}$ (column $k$ of $W_2$). Then $W_2W_1=\sum_kq_kp_k^\top$ and $\langle\Sigma,W_2W_1\rangle=\sum_kq_k^\top\Sigma p_k$. The quadratic form splits over hidden units, and for one unit, $-q^\top\Sigma p=\tfrac12z^\top Hz$ with $z=(p,q)$ and
$$
H=-\begin{pmatrix}0&\Sigma^\top\\\Sigma&0\end{pmatrix}.
$$
The block matrix has eigenvalues $\pm s_\alpha$ with eigenvectors $(v_\alpha,\pm u_\alpha)/\sqrt2$, and zeros for the remaining $n_x+n_y-2r$ dimensions. So $H$ has eigenvalue $-s_\alpha$ along $(v_\alpha,u_\alpha)/\sqrt2$ (input and output weights growing together along the mode, which lowers the loss) and $+s_\alpha$ along $(v_\alpha,-u_\alpha)/\sqrt2$. With $h$ hidden units each $\pm s_\alpha$ appears $h$ times. The origin is a strict saddle with $rh$ downhill directions.

**2.** Let $e$ be the spare hidden direction, so $W_1^\top e=0$ and $W_2e=0$. Perturb as in the hint. Then $W_2W_1\to\Sigma_S+\epsilon^2u_\beta v_\beta^\top$, because the cross terms $\epsilon W_2e\,v_\beta^\top$ and $\epsilon u_\beta e^\top W_1$ vanish. So $E\to\Sigma_{S^c}-\epsilon^2u_\beta v_\beta^\top$ and
$$
L=\tfrac12\|\Sigma_{S^c}\|^2-\epsilon^2s_\beta+O(\epsilon^4).
$$
The perturbation has length $\sqrt2\,\epsilon$, so along the unit direction the curvature is $2\cdot(-\epsilon^2s_\beta)/(\sqrt2\epsilon)^2=-s_\beta<0$: a strict saddle. (If the width is used up, the escape direction instead rotates a hidden unit from a weaker kept mode to a stronger skipped one; Achour et al.'s Theorem 7 covers that case.)

**3.** At depth 3, $L=\text{const}-\langle\Sigma,W_3W_2W_1\rangle+O(\|W\|^6)$. The first non-constant term is cubic, so every second derivative at the origin vanishes and the Hessian is zero. The cubic takes both signs, so the origin is a saddle with no negative curvature: non-strict.
"""
    ),
    markdown(
        r"""
**Code: look at the Hessians.** The cell computes the full Hessian with torch at the origin
for $N=2$ and $N=3$, and at a global minimum for $N=2$, with $n_x=n_y=h=3$ and
$s=(3,2,1)$. Compare the spectra with Exercises 3.2 and 3.3.
"""
    ),
    code(
        r"""
import torch
torch.set_default_dtype(torch.float64)
n, svals = 3, [3.0, 2.0, 1.0]
Sig_np, Un, Vn = make_target(svals, n=n, seed=3)
Sig = torch.tensor(Sig_np)

def loss_flat(theta, N):
    Ws = list(theta.reshape(N, n, n))          # all layers n x n
    W = Ws[0]
    for M in Ws[1:]:
        W = M @ W
    return 0.5 * ((Sig - W) ** 2).sum()

def spectrum(theta, N):
    H = torch.autograd.functional.hessian(lambda th: loss_flat(th, N), theta)
    return np.sort(np.linalg.eigvalsh(H.numpy()))

np.set_printoptions(precision=3, suppress=True)
print("N=2, origin:     ", spectrum(torch.zeros(2 * n * n), 2))
print("N=3, origin:     ", spectrum(torch.zeros(3 * n * n), 3))
# a balanced global minimum for N = 2: W1 = (Sigma^T Sigma)^(1/4) = V S^(1/2) V^T, W2 = U S^(1/2) V^T
W1s = Vn[:, :n] @ np.diag(np.sqrt(svals)) @ Vn[:, :n].T
W2s = Un[:, :n] @ np.diag(np.sqrt(svals)) @ Vn[:, :n].T
theta_min = torch.tensor(np.concatenate([W1s.ravel(), W2s.ravel()]))
ev = spectrum(theta_min, 2)
print("N=2, a minimum:  ", ev)
print(f"   zero eigenvalues (|ev| < 1e-9): {np.sum(np.abs(ev) < 1e-9)}  (gauge dimension n^2 = {n * n})")
"""
    ),
    # ================================================================== §4
    markdown(
        r"""
<a id="td1-s4"></a>

## 4. The end-to-end flow and its kernel

How does the end-to-end map $W=W_2W_1$ move? As $\tau\dot W=K_t[E]$, a weight-dependent
linear operator applied to the error $E=\Sigma-W$ (with $\Sigma=\mathbb E[yx^\top]$, $\tau=1/\eta$).
$K_t$ is the network's neural tangent kernel (NTK); whether it moves is the lazy-versus-rich
question. With balanced layers it is zero on directions not yet learned. In general each
singular mode carries one conserved number $\Delta$, counting layer imbalance and
misalignment, which fixes how far that mode's kernel can move.

**Papers:** Arora, Cohen & Hazan (2018) for the balanced end-to-end equation; Tu, Aranguri
and Jacot (2024) prove the matrix form for wide Gaussian initializations (see
[Readings](#td1-readings)). General wide-network NTK: the Physics of Deep Learning day's [Notebook 2]("""
        + PHYS
        + r"""/02_ntk_limit_kernel_learners_solutions.ipynb) and §7.
"""
    ),
    markdown(
        r"""
<a id="td1-s4-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* two layers $W=W_2W_1$, whitened inputs, $\Sigma=\sum_\alpha s_\alpha u_\alpha v_\alpha^\top$,
$E=\Sigma-W$, $\tau=1/\eta$; the flow (1.1) is $\tau\dot W_1=W_2^\top E$, $\tau\dot W_2=EW_1^\top$, with
conserved imbalance $D=W_2^\top W_2-W_1W_1^\top$. An aligned mode (§1.3) has input weight $b$,
output weight $c$, strength $a=bc$, imbalance $d=c^2-b^2$, and obeys
$\tau\dot a=\sqrt{4a^2+d^2}\,(s-a)$.

**4.1 The end-to-end flow.** From (1.1),

$$
\tau\dot W=W_2W_2^\top\,E+E\,W_1^\top W_1=:K_t[E]. \tag{4.1}
$$

$K_t$ is a positive semi-definite linear map on matrices and $\tau\dot L=-\langle E,K_t[E]\rangle$
(Exercise 4.1). It is the NTK, $\Theta(x,x')=\nabla_\theta f(x)\,\nabla_\theta f(x')^\top$, of
$f(x)=W_2W_1x$ averaged over whitened inputs (Exercise 4.2). A constant $K$ would give kernel
regression, $W(t)=\Sigma-e^{-Kt/\tau}[\Sigma-W(0)]$. Per mode, let $p_\alpha:=W_1v_\alpha$ and
$q_\alpha:=W_2^\top u_\alpha$ (how the hidden layer reads input direction $v_\alpha$ and writes output
direction $u_\alpha$). Then $a_\alpha=p_\alpha\cdot q_\alpha$, and the kernel on the mode is
$k_\alpha:=\langle u_\alpha v_\alpha^\top,K_t[u_\alpha v_\alpha^\top]\rangle=|p_\alpha|^2+|q_\alpha|^2$, which equals
$\sqrt{4a_\alpha^2+d_\alpha^2}$ for an aligned mode.

**4.2 Balanced: the kernel is a function of what has been learned.** If $D=0$, then
$W_2W_2^\top=(WW^\top)^{1/2}$ and $W_1^\top W_1=(W^\top W)^{1/2}$ (Exercise 4.3), so

$$
\tau\dot W=(WW^\top)^{1/2}E+E\,(W^\top W)^{1/2}.
$$

On an aligned mode the kernel is $2a_\alpha$: zero on unlearned directions, large on learned
ones, a minimal form of feature learning. At depth $N$, Arora, Cohen and Hazan find
$\tau\dot W=\sum_{j=1}^N(WW^\top)^{(N-j)/N}\,E\,(W^\top W)^{(j-1)/N}$, which for $N>2$ and a single
output is not the gradient of any function of $W$.

**4.3 Imbalanced: one conserved number per mode.** If different modes use orthogonal hidden
directions (the flow preserves this, Exercise 4.4), each mode obeys $\tau\dot p=q(s-a)$,
$\tau\dot q=p(s-a)$, hence $\tau\dot a=k(s-a)$ and $\tau\dot k=4a(s-a)$, so

$$
\Delta^2:=k^2-4a^2\quad\text{is conserved, and}\quad \boxed{\;\tau\dot a=\sqrt{4a^2+\Delta^2}\;(s-a)\;} \tag{4.3}
$$

The mode's **invariant** $\Delta$ replaces $d$ in (1.4):

$$
\Delta^2=\big(|q|^2-|p|^2\big)^2+4\big(|p|^2|q|^2-(p\cdot q)^2\big)=\big\|\,qq^\top-pp^\top\big\|_*^2,
$$

the squared trace norm of the mode's part of $D$. It counts imbalance *and* misalignment
($\Delta=|d|$ if $p\parallel q$; $\Delta=|p|^2+|q|^2$ if $|p|=|q|$ and $p\perp q$). Since
$\Delta^2=k(0)^2-4a(0)^2$, **$\Delta$ is the initial kernel on the mode, minus the part already
explained by the initial strength.** For i.i.d. Gaussian weights of variance $\sigma^2$ at width
$h\gg n_x,n_y$, the layers have equal norms but $p\perp q$ nearly, so $\Delta\approx2\sigma^2h$
(Exercise 4.4.5). Cross-mode overlaps are then a relative $O(h^{-1/2})$, so (4.3) is
approximate: in this notebook's examples $\Delta$ drifts by tens of percent at $h=8$ and under
1% at $h=2000$, and §5 checks that (4.3) still predicts each mode well. Tu, Aranguri and
Jacot's matrix formula, in our units
$\tau\dot W\approx\sqrt{WW^\top+\sigma^4h^2I}\;E+E\;\sqrt{W^\top W+\sigma^4h^2I}$, gives (4.3) with
$\Delta=2\sigma^2h$ on each mode. For exact solutions when $D=\lambda I$ see
[Dominé et al.](https://arxiv.org/abs/2409.14623); for the same conserved quantities with
per-layer learning rates, [Kunin et al.](https://arxiv.org/abs/2406.06158)

<a id="td1-ex4-1"></a>

**Exercise 4.1 — The end-to-end flow.** Derive (4.1), show $\tau\dot L=-\langle E,K_t[E]\rangle\le0$, and show that $K_t$ is self-adjoint and positive semi-definite in the Frobenius inner product.
"""
    ),
    answer(
        r"""
$\tau\dot W=\tau\dot W_2\,W_1+W_2\,\tau\dot W_1=EW_1^\top W_1+W_2W_2^\top E$.

$L=\tfrac12\|E\|^2+\text{const}$ and $\dot E=-\dot W$, so $\tau\dot L=-\langle E,\tau\dot W\rangle=-\langle E,K_t[E]\rangle$.

For any $X,Y$: $\langle X,K_t[Y]\rangle=\operatorname{tr}(X^\top W_2W_2^\top Y)+\operatorname{tr}(X^\top YW_1^\top W_1)$. Both terms are symmetric under $X\leftrightarrow Y$ (transpose inside the trace and use the symmetry of $W_2W_2^\top$ and $W_1^\top W_1$), so $K_t$ is self-adjoint. With $X=Y=E$, $\langle E,K_t[E]\rangle=\|W_2^\top E\|_F^2+\|EW_1^\top\|_F^2\ge0$. In vectorized form, $K_t=W_1^\top W_1\otimes I+I\otimes W_2W_2^\top$ (column-stacking vec), a sum of two PSD Kronecker products.
"""
    ),
    markdown(
        r"""
<a id="td1-ex4-2"></a>

**Exercise 4.2 — This is the NTK** *(optional)*. For $f(x)=W_2W_1x$, the tangent kernel is the $n_y\times n_y$ matrix $\Theta(x,x')=\sum_\theta\partial_\theta f(x)\,\partial_\theta f(x')^\top$, summed over all weights.

1. Show $\Theta(x,x')=(x\cdot x')\,W_2W_2^\top+(x^\top W_1^\top W_1x')\,I$.
2. Gradient flow moves the function as $\tau\dot f(x)=\mathbb E_{x'}\big[\Theta(x,x')\,(y'-f(x'))\big]$. Show that with whitened inputs this is $\tau\dot Wx=K_t[E]\,x$, i.e. (4.1).

<details><summary>Hint</summary>

Split the sum over $\theta$ into the $W_2$ entries ($\partial f/\partial(W_2)_{ik}=e_i(W_1x)_k$) and the $W_1$ entries ($\partial f/\partial(W_1)_{kj}=W_2e_k\,x_j$).
</details>
"""
    ),
    answer(
        r"""
**1.** $W_2$ entries: $\sum_{i,k}e_i(W_1x)_k\,(W_1x')_ke_i^\top=(x^\top W_1^\top W_1x')\,I$. $W_1$ entries: $\sum_{k,j}(W_2e_k)x_j\,x'_j(W_2e_k)^\top=(x\cdot x')\,W_2\big(\sum_ke_ke_k^\top\big)W_2^\top=(x\cdot x')\,W_2W_2^\top$. Add them.

**2.** $\mathbb E_{x'}\big[(x\cdot x')\,W_2W_2^\top(y'-Wx')\big]=W_2W_2^\top\,\mathbb E\big[(y'-Wx')x'^\top\big]x=W_2W_2^\top(\Sigma-W)x$, using $\mathbb E[y'x'^\top]=\Sigma$ and $\mathbb E[x'x'^\top]=I$. Similarly $\mathbb E_{x'}\big[(x^\top W_1^\top W_1x')(y'-Wx')\big]=\mathbb E[(y'-Wx')x'^\top]\,W_1^\top W_1x=EW_1^\top W_1x$. So $\tau\dot f(x)=K_t[E]\,x$ for every $x$, i.e. $\tau\dot W=K_t[E]$. The kernel of kernel regression and the operator $K_t$ are the same object, seen in function space and in matrix space.
"""
    ),
    markdown(
        r"""
<a id="td1-ex4-3"></a>

**Exercise 4.3 — Balanced kernels.**

1. Show that $D=0$ implies $W_2W_2^\top=(WW^\top)^{1/2}$ and $W_1^\top W_1=(W^\top W)^{1/2}$.
2. Show that on an aligned mode the kernel equals $2a$, and recover $\tau\dot a=2a(s-a)$ from (4.1).
3. *(Optional.)* Check that the Arora–Cohen–Hazan formula gives $Na^{2-2/N}$ on an aligned mode.

<details><summary>Hint</summary>

Use $D=0$ to replace $W_1W_1^\top$ by $W_2^\top W_2$ inside $WW^\top$. A PSD matrix has a unique PSD square root.
</details>
"""
    ),
    answer(
        r"""
**1.** $WW^\top=W_2W_1W_1^\top W_2^\top=W_2(W_2^\top W_2)W_2^\top=(W_2W_2^\top)^2$, using $W_1W_1^\top=W_2^\top W_2$. A PSD matrix has a unique PSD square root, and $W_2W_2^\top$ is PSD, so $W_2W_2^\top=(WW^\top)^{1/2}$. Likewise $W^\top W=W_1^\top(W_2^\top W_2)W_1=(W_1^\top W_1)^2$.

**2.** On an aligned mode $W=\sum_\alpha a_\alpha u_\alpha v_\alpha^\top$ with $a_\alpha\ge0$, so $(WW^\top)^{1/2}u_\alpha=a_\alpha u_\alpha$ and $(W^\top W)^{1/2}v_\alpha=a_\alpha v_\alpha$. Then $u_\alpha^\top K_t[E]v_\alpha=a_\alpha(s_\alpha-a_\alpha)+(s_\alpha-a_\alpha)a_\alpha=2a_\alpha(s_\alpha-a_\alpha)$, which is $\tau\dot a_\alpha$. The kernel on mode $\alpha$ is $2a_\alpha$.

**3.** $(WW^\top)^{(N-j)/N}u=a^{2(N-j)/N}u$ and $(W^\top W)^{(j-1)/N}v=a^{2(j-1)/N}v$, so the $j$-th term contributes $a^{2(N-1)/N}(s-a)$, and the sum over $j=1..N$ gives $Na^{2-2/N}(s-a)$, which is (2.1).
"""
    ),
    markdown(
        r"""
<a id="td1-ex4-4"></a>

**Exercise 4.4 — The per-mode invariant $\Delta$.**

1. Show that $\tau\dot p_\alpha=W_2^\top Ev_\alpha$ and $\tau\dot q_\alpha=W_1E^\top u_\alpha$. Show that if $u_\beta^\top Wv_\alpha=0$ for all $\beta\ne\alpha$ (the error is diagonal in the $U,V$ basis), these become $\tau\dot p_\alpha=q_\alpha(s_\alpha-a_\alpha)$ and $\tau\dot q_\alpha=p_\alpha(s_\alpha-a_\alpha)$.
2. Show that if the off-diagonal overlaps $q_\beta\cdot p_\alpha$, $p_\beta\cdot p_\alpha$, $q_\beta\cdot q_\alpha$ ($\beta\ne\alpha$) all vanish initially, they vanish for all time, so the assumption in 1 is self-consistent.
3. Derive $\tau\dot a=k(s-a)$ and $\tau\dot k=4a(s-a)$, deduce that $\Delta^2=k^2-4a^2$ is conserved, and derive (4.3).
4. Show $\Delta^2=(|q|^2-|p|^2)^2+4(|p|^2|q|^2-(p\cdot q)^2)$, and that this is the square of the trace norm of $qq^\top-pp^\top$.
5. For i.i.d. $\mathcal N(0,\sigma^2)$ entries in both layers, estimate $|p|^2$, $|q|^2$, $p\cdot q$ and $\Delta$ for large $h$. What if instead $W_1$ has variance $1/n_x$ and $W_2$ variance $1/h$ (fan-in initialization)?

<details><summary>Hint</summary>

For 4, $qq^\top-pp^\top$ has rank at most 2, trace $|q|^2-|p|^2$, and on the plane spanned by $p,q$ its determinant is $-(|p|^2|q|^2-(p\cdot q)^2)\le0$, so its two eigenvalues have opposite signs. For 5, $p=W_1v$ has i.i.d. $\mathcal N(0,\sigma^2)$ entries because $v$ is a unit vector.
</details>
"""
    ),
    answer(
        r"""
**1.** $\tau\dot p_\alpha=\tau\dot W_1v_\alpha=W_2^\top Ev_\alpha$ and $\tau\dot q_\alpha=(\tau\dot W_2)^\top u_\alpha=W_1E^\top u_\alpha$. If $W$ is diagonal in the $U,V$ basis, then so is $E=\Sigma-W$, and $Ev_\alpha=(s_\alpha-a_\alpha)u_\alpha$, $E^\top u_\alpha=(s_\alpha-a_\alpha)v_\alpha$. So $\tau\dot p_\alpha=(s_\alpha-a_\alpha)W_2^\top u_\alpha=(s_\alpha-a_\alpha)q_\alpha$, and similarly for $q_\alpha$. (Directions of $\Sigma$'s null spaces are included as modes with $s=0$.)

**2.** In general, $\tau\dot p_\alpha=\sum_\gamma q_\gamma E_{\gamma\alpha}$ and $\tau\dot q_\alpha=\sum_\gamma p_\gamma E_{\alpha\gamma}$, where $E_{\gamma\alpha}=u_\gamma^\top Ev_\alpha=s_\alpha\delta_{\gamma\alpha}-q_\gamma\cdot p_\alpha$. Differentiating an off-diagonal overlap, e.g. $\tau\frac{d}{dt}(q_\beta\cdot p_\alpha)=\sum_\gamma(p_\gamma\cdot p_\alpha)E_{\beta\gamma}+\sum_\gamma(q_\beta\cdot q_\gamma)E_{\gamma\alpha}$, every term on the right contains at least one off-diagonal factor: an overlap between different modes, or an off-diagonal entry $E_{\gamma\alpha}=-q_\gamma\cdot p_\alpha$. (Terms with $\gamma\notin\{\alpha,\beta\}$ contain two.) So if $x$ collects all off-diagonal overlaps, $\dot x=F(x,\dots)$ with $F=0$ at $x=0$, and by uniqueness $x(0)=0$ implies $x\equiv0$. For small nonzero $x(0)$, as at random initialization, §5 checks numerically that $x$ stays small.

**3.** $\tau\dot a=\tau(\dot p\cdot q+p\cdot\dot q)=(|q|^2+|p|^2)(s-a)=k(s-a)$, and $\tau\dot k=2\tau(p\cdot\dot p+q\cdot\dot q)=4(p\cdot q)(s-a)=4a(s-a)$. Then $\tau\frac{d}{dt}(k^2-4a^2)=2k\cdot4a(s-a)-8a\cdot k(s-a)=0$. With $k\ge0$, $k=\sqrt{4a^2+\Delta^2}$, which gives (4.3).

**4.** $k^2-4a^2=(|p|^2+|q|^2)^2-4(p\cdot q)^2=(|q|^2-|p|^2)^2+4|p|^2|q|^2-4(p\cdot q)^2$. For the trace norm: $M=qq^\top-pp^\top$ acts as zero off the plane of $p,q$. On that plane it has trace $t=|q|^2-|p|^2$ and determinant $\delta=-(|p|^2|q|^2-(p\cdot q)^2)\le0$ (Cauchy–Schwarz), so its eigenvalues $\lambda_\pm$ have opposite signs and $\|M\|_*=\lambda_+-\lambda_-=\sqrt{t^2-4\delta}=\Delta$. If $p\parallel q$ the determinant vanishes and $\Delta=\big||q|^2-|p|^2\big|=|d|$.

**5.** $p=W_1v$ has $h$ i.i.d. $\mathcal N(0,\sigma^2)$ entries, so $|p|^2=\sigma^2h\,(1+O(h^{-1/2}))$; the same for $q$. $p$ and $q$ are independent, so $p\cdot q$ has mean 0 and standard deviation $\sigma^2\sqrt h$. Hence $a(0)=O(\sigma^2\sqrt h)$, $k(0)\approx2\sigma^2h$, and $\Delta=\sqrt{k^2-4a^2}\approx2\sigma^2h$. Cross-mode overlaps are also $O(\sigma^2\sqrt h)$, a relative $O(h^{-1/2})$, so modes separate approximately at large width. With fan-in initialization, $|p|^2\approx h/n_x$ and $|q|^2\approx1$, so $\Delta\approx h/n_x+1$, growing with width.
"""
    ),
    # ================================================================== §5
    markdown(
        r"""
<a id="td1-s5"></a>

## 5. Rich, lazy and mixed

When is training lazy (a frozen kernel, as in kernel regression) and when rich (a kernel
that grows with what is learned)? Here both are settings of one dial: each mode's conserved
invariant $\Delta$ (§4) against its target singular value $s$. Small $\Delta$ gives
saddle-to-saddle, strongest-first learning; large $\Delta$ freezes the kernel; in between,
strong modes are rich and weak ones lazy. "Large initialization is lazy" holds for large
*Gaussian* weights, which are strongly imbalanced, and fails for large *balanced* ones.

**Papers:** Tu, Aranguri & Jacot (2024) for the phase diagram; Chizat, Oyallon & Bach (2019)
on lazy training (see [Readings](#td1-readings)).
"""
    ),
    markdown(
        r"""
<a id="td1-s5-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* as in §4: a mode with singular value $s$ has strength $a=u^\top Wv$, kernel
$k=|p|^2+|q|^2$ ($p=W_1v$, $q=W_2^\top u$), invariant $\Delta^2=k(0)^2-4a(0)^2$, and obeys (4.3),
$\tau\dot a=\sqrt{4a^2+\Delta^2}\,(s-a)$: exactly for separated modes, approximately for wide
random initializations. $\Delta=|c^2-b^2|$ for an aligned mode with input weight $b$ and output
weight $c$; $\Delta\approx2\sigma^2h$ for Gaussian weights of variance $\sigma^2$ at width $h$.
$\Sigma_S=\sum_{\alpha\in S}s_\alpha u_\alpha v_\alpha^\top$; $\tau=1/\eta$.

**5.1 Rich: saddle to saddle.** If $\Delta_\alpha\ll s_\alpha$ and $|a_\alpha(0)|\ll s_\alpha$ for every
mode, each follows a near-logistic with half-time $t_\alpha\approx(\tau/2s_\alpha)\ln(4s_\alpha/\Delta_\alpha)$,
like a balanced mode with $a_0=\Delta/4$ (Exercise 5.2; this matches §1.5's seed at large
width). So $W(t)$ passes near $\Sigma_{\{1\}}$, then $\Sigma_{\{1,2\}}$, and so on, the saddles of
§3: a staircase loss, effective rank growing one step at a time, and the kernel on mode
$\alpha$ growing from $\Delta_\alpha$ to $2s_\alpha$. The low-rank bias is in the *trajectory*: the
optimum is $\Sigma$ whatever the path, so the bias shows only under early stopping (a
near-hard threshold on the singular values, where a frozen isotropic kernel shrinks every
mode by the same factor) or where the minimizer is not unique (matrix completion, sensing).
Li, Luo and Lyu prove that the first phase is greedy rank-one learning.

**5.2 Lazy: a frozen kernel.** If $\max(|a_0|,s)\ll\Delta$, the kernel $\sqrt{4a^2+\Delta^2}$
changes by a relative amount of at most about $2\max(a_0^2,s^2)/\Delta^2$ (Exercise 5.1), and
$\tau\dot a\approx\Delta(s-a)$: every mode relaxes at rate $\Delta/\tau$, with no plateau and no
ordering. This is kernel regression with the initial NTK, and the mode's weights move by a
relative $O(s/\Delta)$. Examples: one large layer ($|d|\gg s$), Gaussian weights with
$\sigma^2h\gg s$, and fan-in initialization at large width ($\Delta\approx h/n_x$). The stable step
is then about $2/\Delta$, so a larger kernel does not reduce the number of steps. This is
Chizat, Oyallon and Bach's point: a large output scale grows the kernel while the needed
function change stays fixed.

**5.3 Large balanced weights are not lazy.** An aligned balanced mode ($\Delta=0$) started at
$a_0\gg s$ has kernel $2a$, which falls from $2a_0$ to $2s$, a relative change of $1-s/a_0$.
Balanced layers make the function the *square* of the per-layer scale, so large balanced
weights mean a large initial function, and removing the excess is an $O(1)$ change of the
weights that carry it: 5.2's criterion fails through $|a_0|$. Large Gaussian initialization
is lazy for a different reason: its input and output vectors are nearly orthogonal, which
makes $\Delta$ large while $a_0$ stays small.

**5.4 Mixed.** If $\Delta$ sits between the singular values, modes with $s_\alpha\gg\Delta/2$ are
rich (kernel growing by $\sim2s_\alpha/\Delta$, along an S-shaped curve when $s>\sqrt2\,\Delta$) and
modes with $s_\alpha\ll\Delta/2$ are lazy (rate $\Delta/\tau$). The rich modes finish *first*, since
$(\tau/2s)\ln(4s/\Delta)\ll\tau/\Delta$ when $s\gg\Delta$. For Gaussian initialization the threshold
$\Delta/2\approx\sigma^2h$ is Tu, Aranguri and Jacot's $\sigma^2w$ (their width $w$ is our $h$).
Their phase diagram, for an $n\times n$ low-rank target with singular values of order $n$, is
lazy when $\sigma^2w\gg n$ and active or mixed when $\sigma^2w\ll n$; the mixed regime recovers a
low-rank target from noisy data where the lazy regime fits the noise, at the cost of a
longer time. The mode explorer's "large balanced init" and "mixed" presets show 5.3 and 5.4.

<a id="td1-ex5-1"></a>

**Exercise 5.1 — When is the kernel frozen?**

1. For $\max(|a_0|,s)\ll\Delta$, bound the relative change of the mode kernel along the trajectory, and estimate how much the mode's weights move relative to their joint size $\sqrt k$.
2. For an aligned balanced mode started at $a_0>s$, compute the relative change of the kernel. Evaluate it for $a_0=5s$.
3. A student says: "Large initialization puts you in the lazy regime." Rewrite the sentence so that it is true for this model.

<details><summary>Hint</summary>

Bound $k=\sqrt{4a^2+\Delta^2}$ between its values at the ends of the trajectory. For the weights, integrate $|\dot p|\le|q|\,|s-a|/\tau$.
</details>
"""
    ),
    answer(
        r"""
**1.** $k=\sqrt{4a^2+\Delta^2}$ with $a$ between $a_0$ and $s$. With $m=\max(|a_0|,s)$, $\Delta\le k\le\sqrt{4m^2+\Delta^2}$, so the relative change is at most $\sqrt{1+4m^2/\Delta^2}-1\approx2m^2/\Delta^2$. For the weights: $k=|p|^2+|q|^2\approx\Delta$. From $\tau\dot p=q(s-a)$ and $\tau\dot q=p(s-a)$, the total motion is $\int(|\dot p|+|\dot q|)\,dt\lesssim\sqrt{2k}\int|s-a|\,dt/\tau\approx\sqrt{2\Delta}\,|s-a_0|/\Delta$, because $\int_0^\infty(s-a)\,dt/\tau=(s-a_0)/\Delta$ for the exponential. Relative to $\sqrt k\approx\sqrt\Delta$ this is $O(|s-a_0|/\Delta)$, which vanishes in the lazy limit (a tiny layer can still change a lot relative to itself).

**2.** $k=2a$ falls from $2a_0$ to $2s$, so the relative change is $(2a_0-2s)/2a_0=1-s/a_0=0.8$ at $a_0=5s$. The kernel is not frozen, although the weights are large.

**3.** "Initialization with a large invariant $\Delta$ on each mode (a large initial kernel not accounted for by the initial function, $\Delta^2=k(0)^2-4a(0)^2$), together with an initial function that is not itself large ($\max(|a_0|,s)\ll\Delta$), puts you in the lazy regime. Large Gaussian initialization, wide networks with fan-in initialization, and one layer much larger than the other all have large $\Delta$. Large balanced, aligned weights have $\Delta=0$ and are not lazy."
"""
    ),
    markdown(
        r"""
<a id="td1-ex5-2"></a>

**Exercise 5.2 — The mixed regime.** Take $a_0=0$ in (4.3).

1. For $s\gg\Delta$, use the closed form of §1.4 (with $d\to\Delta$) to show $t_{1/2}\approx(\tau/2s)\ln(4s/\Delta)$, i.e. the mode behaves like a balanced mode with $a_0=\Delta/4$. By what factor does its kernel grow?
2. For $s\ll\Delta$, show that the mode is lazy and give its time constant.
3. With Gaussian initialization ($\Delta\approx2\sigma^2h$) and singular values $s_1>\dots>s_r$, which modes are rich and which are lazy? Compare with Tu–Aranguri–Jacot's threshold.

<details><summary>Hint</summary>

For 1, expand $R=\sqrt{4s^2+\Delta^2}\approx2s$ and evaluate $F(s/2)-F(0)$ to leading order in $\Delta/s$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $R\approx2s$: $F(s/2)=\frac1R\ln\frac{2s^2+\Delta^2+R\sqrt{s^2+\Delta^2}}{s}\approx\frac{1}{2s}\ln\frac{2s^2+2s^2}{s}=\frac{\ln4s}{2s}$, and $F(0)=\frac1R\ln\frac{\Delta^2+R\Delta}{2s}\approx\frac1{2s}\ln\frac{2s\Delta}{2s}=\frac{\ln\Delta}{2s}$. So $t_{1/2}/\tau\approx\frac1{2s}\ln\frac{4s}{\Delta}$. The balanced half-time is $\frac1{2s}\ln(s/a_0-1)\approx\frac1{2s}\ln(s/a_0)$, so the two agree for $a_0=\Delta/4$. (The math check script confirms this numerically.) The kernel grows from $\Delta$ to $\sqrt{4s^2+\Delta^2}\approx2s$, a factor $2s/\Delta\gg1$.

**2.** For $s\ll\Delta$, Exercise 5.1.1 applies: $\tau\dot a\approx\Delta(s-a)$, time constant $\tau/\Delta$, independent of $s$.

**3.** Modes with $s_\alpha\gg\sigma^2h$ are rich, learned in order of decreasing $s_\alpha$ at times $\approx(\tau/2s_\alpha)\ln(2s_\alpha/(\sigma^2h))$. Modes with $s_\alpha\ll\sigma^2h$ are lazy, all relaxing at rate $2\sigma^2h/\tau$. The strong rich modes still finish first: $(\tau/2s)\ln(4s/\Delta)\ll\tau/\Delta$ when $s\gg\Delta$. Only modes with $s>\sqrt2\,\Delta$ have a sigmoidal (convex) rise at all: with $f(a)=\sqrt{4a^2+\Delta^2}(s-a)$, $\ddot a\propto f'(a)f(a)$ and $f'(a)=(4as-8a^2-\Delta^2)/\sqrt{4a^2+\Delta^2}$, which is positive for some $a$ exactly when $8a^2-4sa+\Delta^2=0$ has real roots, i.e. $s>\sqrt2\,\Delta$. The crossover is at $s\approx\Delta/2=\sigma^2h$. Tu–Aranguri–Jacot's formula gives the per-mode rate $2\sqrt{s^2+\sigma^4w^2}$ and the same threshold $s\approx\sigma^2w$.
"""
    ),
    markdown(
        r"""
**Code: the invariant $\Delta$ for five initializations.** Initial $a(0)$, $k(0)$ and $\Delta$ on
the four modes of a target with $s=(3,2,1,0.5)$, for: small Gaussian ($h=8$,
$\sigma=10^{-3}$); large imbalance ($W_2W_2^\top=30I$, $W_1$ tiny); wide Gaussian ($h=2000$,
$\sigma^2h=20$); and large *balanced* weights ($D=0$, every singular value of $W(0)$ equal to
15, so the initial kernel is 30, as in the lazy runs), aligned with $\Sigma$ or random. The
last two columns are the largest cross-mode overlap relative to $k(0)$ (the per-mode picture
needs it small) and the operator norm of $D$.
"""
    ),
    code(
        r"""
svals = [3.0, 2.0, 1.0, 0.5]
Sigma, U, V = make_target(svals)

def inits(seed=0):
    r = np.random.default_rng(seed)
    out = {}
    out["small Gaussian"] = (1e-3 * r.normal(size=(8, 6)), 1e-3 * r.normal(size=(6, 8)))
    Q, _ = np.linalg.qr(r.normal(size=(8, 8)))
    out["large imbalance"] = (1e-3 * r.normal(size=(8, 6)), np.sqrt(30.0) * Q[:6, :])   # W2 W2^T = 30 I
    h = 2000; sig2 = 20 / h
    out["wide Gaussian"] = (np.sqrt(sig2) * r.normal(size=(h, 6)), np.sqrt(sig2) * r.normal(size=(6, h)))
    Qb, _ = np.linalg.qr(r.normal(size=(8, 8)))
    root = np.sqrt(15.0) * np.eye(6)
    out["large balanced, aligned"] = (Qb[:, :6] @ root @ V.T, U @ root @ Qb[:, :6].T)      # D = 0, every a(0) = 15
    A = r.normal(size=(6, 6)); Ua, _, Vta = np.linalg.svd(A)
    out["large balanced, random"] = (Qb[:, :6] @ root @ Vta, Ua @ root @ Qb[:, :6].T)      # D = 0, W(0) not aligned
    return out

def mode_stats(W1, W2, k=4):
    P, Q = W1 @ V, W2.T @ U                       # columns: p_alpha, q_alpha (all 6 directions)
    a = np.einsum("ia,ia->a", P, Q)
    kk = (P**2).sum(0) + (Q**2).sum(0)
    Delta = np.sqrt(np.maximum(kk**2 - 4 * a**2, 0))
    off = lambda G: np.abs(G - np.diag(np.diag(G))).max()     # largest off-diagonal entry
    cross = max(off(Q.T @ P), off(P.T @ P), off(Q.T @ Q))    # overlaps between different modes
    return a[:k], kk[:k], Delta[:k], cross / kk[:k].min()

def D_norm(W1, W2):
    # operator norm of the conserved imbalance D = W2^T W2 - W1 W1^T (h x h), computed cheaply:
    # D = A^T J A with A = [W2; W1^T] and J = diag(I, -I).
    A = np.vstack([W2, W1.T]); J = np.diag(np.r_[np.ones(W2.shape[0]), -np.ones(W1.shape[1])])
    _, Rm = np.linalg.qr(A.T)                     # A^T = Q R, so D = Q (R J R^T) Q^T
    return np.abs(np.linalg.eigvalsh(Rm @ J @ Rm.T)).max()

print(f"{'init':24s} {'a(0)':>30s} {'k(0)':>30s} {'Delta':>30s}  cross/k(0)  ||D||")
for name, (W1, W2) in inits().items():
    a, kk, Dl, cr = mode_stats(W1, W2)
    f = lambda x: " ".join(f"{v:7.2g}" for v in x)
    print(f"{name:24s} {f(a):>30s} {f(kk):>30s} {f(Dl):>30s}  {cr:9.2g}  {D_norm(W1, W2):7.2g}")
"""
    ),
    markdown(
        r"""
Small Gaussian: $\Delta\ll s$ (rich). Large imbalance and wide Gaussian: $\Delta\gg s$ (lazy).
Both balanced runs have $D=0$ despite big weights and kernels; for the random one the
cross-mode overlaps are as large as $\Delta$, so its per-mode numbers mean little.

**Code: stagewise versus exponential, and which kernels move.** Training from each of the
five initializations, plus a three-layer network from small Gaussian weights (per-layer
$\sigma=0.1$). Dotted: the prediction of (4.3) from each mode's $a(0)$ and $\Delta$. Printed:
the largest relative change of the mode kernels $k_\alpha$ and of the full operator $K_t$. The
large initializations use $\eta=0.002$ instead of $0.005$. Why?
"""
    ),
    code(
        r"""
def kernel_op(W1, W2):
    # K_t as a (ny*nx) x (ny*nx) matrix acting on row-major vec(E)
    return np.kron(W2 @ W2.T, np.eye(W1.shape[1])) + np.kron(np.eye(W2.shape[0]), W1.T @ W1)

runs = {"small Gaussian": (0.005, 16), "large imbalance": (0.002, 1), "wide Gaussian": (0.002, 1),
        "large balanced, aligned": (0.002, 6), "large balanced, random": (0.002, 6)}
mode_k = lambda W1, W2: ((W1 @ V[:, :4])**2).sum(0) + ((W2.T @ U[:, :4])**2).sum(0)
fig, axs = plt.subplots(2, 3, figsize=(13, 6.4)); axs = axs.ravel()
for ax, (name, (W1, W2)) in zip(axs, inits().items()):
    eta, T = runs[name]
    ts, hist = gd([W1, W2], Sigma, eta, int(T / eta), every=max(1, int(T / eta) // 200))
    A = np.array([mode_strengths(w2 @ w1, U, V, 4) for w1, w2 in hist])
    a0, _, Dl, cross = mode_stats(W1, W2)
    for i in range(4):
        ax.plot(ts, A[:, i], color=f"C{i}", lw=2, label=f"s = {svals[i]}")
        if name != "large balanced, random":                  # per-mode prediction needs separated modes
            ax.plot(ts, mode_ode(svals[i], Dl[i], a0[i], ts), "k:", lw=1)
    K0, K1 = kernel_op(W1, W2), kernel_op(*hist[-1])
    rel_op = np.linalg.norm(K1 - K0, 2) / np.linalg.norm(K0, 2)
    rel = np.max(np.abs(mode_k(*hist[-1]) / mode_k(W1, W2) - 1))
    lab = f"mode-kernel change {100 * rel:.0f}%" if rel < 10 else f"mode kernels grow up to ~{rel:.0e}x"
    ax.set_title(f"{name}\n{lab}", fontsize=9); ax.set_xlabel("t / tau")
    print(f"{name:24s} mode kernels {rel:9.3g}   operator {rel_op:9.3g}   "
          f"final loss {0.5 * np.sum((Sigma - product(hist[-1]))**2):.1e}")
# three layers from small Gaussian initialization
r = np.random.default_rng(5)
Ws3 = [0.1 * r.normal(size=(8, 6)), 0.1 * r.normal(size=(8, 8)), 0.1 * r.normal(size=(6, 8))]
ts, hist = gd(Ws3, Sigma, 0.005, int(16 / 0.005), every=20)
A = np.array([mode_strengths(product(w), U, V, 4) for w in hist])
for i in range(4):
    axs[5].plot(ts, A[:, i], color=f"C{i}", lw=2)
axs[5].set_title("three layers, small Gaussian", fontsize=9); axs[5].set_xlabel("t / tau")
axs[0].set_ylabel("mode strength a"); axs[3].set_ylabel("mode strength a"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
print("dotted: prediction of (4.3) from each mode's a(0) and Delta at initialization")
"""
    ),
    markdown(
        r"""
**What you should see.** Small Gaussian: a staircase in order of $s$, predictions within a
few percent (the rest is finite-step error and cross-mode overlap at $h=8$), kernels growing
by orders of magnitude. Large imbalance and wide Gaussian: all modes relax together at rate
$\Delta\approx30$–$40$, as predicted, with kernel changes of 1–2%. Large balanced, aligned: the
same initial kernel ($k=30$) but $\Delta=0$; each mode decays along the logistic from
$a_0=15$, its kernel falling by $1-s/15$ (80–97%); the leftover loss is the two $s=0$
directions, decaying like $1/(2t)$. Large balanced, random: a similar change. Three layers:
the same stagewise ordering (no prediction overlaid).
"""
    ),
    answer(
        r"""
**Why a smaller $\eta$ for the large initializations.** Near a frozen kernel, the parameter-space curvature is the Gauss–Newton term, whose nonzero eigenvalues are the kernel's. With kernel eigenvalues around $30$–$40$ (lazy runs), gradient descent needs $\eta\lesssim2/40$ to be stable, and $\eta k\ll1$ for the discrete trajectory to follow the flow. For the large balanced runs the residual term of the Hessian adds to this (for a balanced scalar mode at $a=15$, $s=3$ the Hessian eigenvalues are $42$ and $-12$, against a kernel of $30$), so a small step is needed there too. Raising the overall kernel scale buys no speed in steps once $\eta$ is scaled down to match.
"""
    ),
    markdown(
        r"""
**Code: the mixed regime.** Wide Gaussian initialization with $\sigma^2h=0.5$ (so
$\Delta\approx1$), and a target with singular values on both sides of the threshold
$\Delta/2\approx0.5$. Left: $a/s$ for each mode, with the predictions of (4.3). Right: each
mode's kernel relative to its initial value.
"""
    ),
    code(
        r"""
svals_m = [8.0, 3.0, 0.25, 0.05]
Sigma_m, Um, Vm = make_target(svals_m, seed=7)
r = np.random.default_rng(8); h = 2000; sig2 = 0.5 / h
W1, W2 = np.sqrt(sig2) * r.normal(size=(h, 6)), np.sqrt(sig2) * r.normal(size=(6, h))
P, Q = W1 @ Vm[:, :4], W2.T @ Um[:, :4]
a0 = np.sum(P * Q, 0); k0 = (P**2).sum(0) + (Q**2).sum(0); Dl = np.sqrt(k0**2 - 4 * a0**2)
ts, hist = gd([W1, W2], Sigma_m, 0.002, 2500, every=25)
A = np.array([mode_strengths(w2 @ w1, Um, Vm, 4) for w1, w2 in hist])
Kmode = np.array([np.sum((w1 @ Vm[:, :4])**2, 0) + np.sum((w2.T @ Um[:, :4])**2, 0) for w1, w2 in hist])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for i, s in enumerate(svals_m):
    axs[0].plot(ts, A[:, i] / s, color=f"C{i}", lw=2, label=f"s = {s}")
    axs[0].plot(ts, mode_ode(s, Dl[i], a0[i], ts) / s, "k:", lw=1)
    axs[1].plot(ts, Kmode[:, i] / Kmode[0, i], color=f"C{i}", lw=2)
axs[0].set_title(f"a / s   (Delta = {Dl.mean():.2f})"); axs[0].legend(fontsize=8)
axs[1].set_title("kernel on each mode, relative to t = 0")
for ax in axs: ax.set_xlabel("t / tau")
plt.tight_layout(); plt.show()
print("kernel growth factors k(end)/k(0):", np.round(Kmode[-1] / Kmode[0], 2),
      " predicted sqrt(4s^2+Delta^2)/Delta:", np.round(np.sqrt(4 * np.array(svals_m)**2 + Dl**2) / Dl, 2))
"""
    ),
    markdown(
        r"""
The two large modes are rich: their kernels grow by factors near $2s/\Delta$ and they rise
along S-shaped curves, the strongest first. The two small modes are lazy: their kernels
barely move and they relax at the common rate $\Delta$, finishing *after* the rich modes.
The predictions of (4.3) lie on top of the simulations.
"""
    ),
    # ================================================================== §6
    markdown(
        r"""
<a id="td1-s6"></a>

## 6. Which heuristics are derived, and which are planted

Several widely quoted lessons come from this model. Which follow from the mathematics of
§1–§5, and which from structure put into the data or the initialization? In short ($s$ is a
singular value of the input–output correlation $\mathbb E[yx^\top]$):

| Heuristic | Derived in this model | Planted by the setup |
|---|---|---|
| 1. Stagewise learning, ordered by $s$ | from small initialization, strongest modes first, each after a plateau | small initialization; gaps in the spectrum; whitened inputs, so that the stages are the SVD of $\mathbb E[yx^\top]$ |
| 2. Semantic development, coarse to fine | the timing, the sigmoidal shape, deep versus shallow | tree-structured data whose coarse splits carry the largest $s$ |
| 3. Low-rank, saddle-to-saddle learning | the path visits partial reconstructions in order | small initialization; it decides the endpoint only with early stopping or non-unique minimizers |
| 4. Orthogonal initialization | no small-kernel plateau at the start; but half of square orthogonal starts stall | positively aligned $O(1)$ modes and a per-depth learning rate, assumed in the argument |
| 5. Generalization dynamics ($n\approx p$ overtraining) | nothing about depth: these are properties of linear least squares | i.i.d. isotropic inputs |
| 6. Small initialization is rich, large is lazy | the dial is imbalance against what is left to learn | "large" must mean imbalanced; large balanced weights are not lazy |

**Paper:** Saxe, McClelland & Ganguli (2019) on semantic development (see
[Readings](#td1-readings)).
"""
    ),
    markdown(
        r"""
<a id="td1-s6-walkthrough"></a>

**Derivations and walkthrough.** The table above, with reasons. **Derived** means it follows from §1–§5 (linear network, square
loss, whitened inputs); **planted** names what the headline needs from the data or the
initialization; **transfer** is the evidence beyond linear networks. Notation:
$\Sigma_{xx}=\mathbb E[xx^\top]$, $\Sigma_{yx}=\mathbb E[yx^\top]$ with singular values $s$; $a_0$ the
initial mode strength; $\Delta$ the per-mode invariant of §4 ($\approx2\sigma^2h$ for Gaussian
weights of variance $\sigma^2$ at width $h$); $\tau=1/\eta$.

**1. Stagewise learning.** *Derived:* from small initialization, modes are learned in order of
decreasing $s$, each after a plateau $\approx(\tau/2s)\ln(4s/\Delta)$; alignment is derived for
the first mode (§1.5) and checked numerically for the rest. *Planted:* (i) small
initialization ($\Delta\ll s$, else no stages, §5.2); (ii) spectral gaps, since separate steps
need $\delta s/s\gg2\ln9/\ln(1/a_0)\approx4.4/\ln(1/a_0)$ (Exercise 6.1); (iii) whitened inputs,
which make the stages the SVD of $\Sigma_{yx}$ (with $\Sigma_{xx}\ne I$ the modes couple unless
$\Sigma_{xx}$ and $\Sigma_{yx}$ share singular vectors). *Transfer:* roughly stagewise learning
appears in some nonlinear settings (§7), but the "modes" are not an SVD of the data.

**2. Semantic development.** *Derived:* given 1, coarse distinctions come first if they carry
larger $s$, with stage-like transitions, and a shallow network shows no stages. *Planted:*
the data. Saxe, McClelland and Ganguli (2019) generate features by branching diffusion down a
tree (each feature flips sign with probability $\epsilon$ per branch) with one-hot inputs, which
makes the singular vectors the tree's splits with $s$ decreasing down the tree (Exercise
6.2). Whether real item–feature statistics have such a spectrum, the dynamics cannot say.
*Transfer:* a qualitative match to an earlier nonlinear simulation (their Fig. 2).

**3. Low-rank-first learning.** *Derived:* with $\Delta\ll s$ the path visits
$\Sigma_{\{1\}},\Sigma_{\{1,2\}},\dots$ (§5.1), a property of initialization scale, factorization and
gradient flow together, not of the loss (the shallow model has the same loss and learns all
modes at once). Proved: the first escape (Jacot et al.) and the first greedy rank-one phase
(Li, Luo, Lyu); the full sequence is conjectured in general. *Planted:* small initialization
and spectral gaps. The optimum is $\Sigma$ anyway, so the bias matters only with early stopping
or underdetermined problems. *Transfer:* §7's alternating gradient flows.

**4. Orthogonal initialization.** *Derived:* every end-to-end singular value starts at 1, the
initial balanced kernel is $N$ times the identity (Exercise 6.3), so there is no small-kernel
plateau, and with $\eta\propto1/N$ (§2.3) the step count need not grow with depth. But with
square layers balanced flow conserves the sign of $\det W$, so half of random orthogonal
starts cannot reach $\Sigma$ and stall near a saddle (Exercise 6.3 and its cell). *Planted:*
nothing in the data; the argument assumes modes starting *positively* aligned with
$a_0=O(1)$, targets of order 1 and a per-depth learning rate, whereas a random orthogonal start
with $n\times n$ layers has mode strengths $O(n^{-1/2})$ of random sign. *Transfer:* the
nonlinear analogue (Jacobian singular values near 1) depends on the nonlinearity; Pennington,
Schoenholz and Ganguli show ReLU networks cannot reach it and orthogonal sigmoidal ones can.

**5. Generalization dynamics.** *Derived elsewhere:* overtraining worst at $n\approx p$
(samples ≈ parameters), early stopping as a spectral filter, and the cost of a large
initialization in directions the data never move are properties of linear least squares,
not of depth. The MDL day's MDL1 derives them, in
[§4 *Regularization*](../notebooks/mdl1_statistical_learning.html#wb2-spectral-filter) (exercise 4.3)
and the keystone [*double descent in least squares*](../notebooks/mdl1_statistical_learning.html#wb2-double-descent)
exercises. *Planted:* Advani, Saxe and Sompolinsky's curves assume i.i.d. isotropic inputs
(hence Marchenko–Pastur; real spectra are closer to power laws) and a rank-one ansatz for deep
scalar-output networks. *Transfer:* simulations, and a model with a fixed random first layer.

**6. Small is rich, large is lazy.** *Derived:* the dial is imbalance against what is left to
learn (§5). *Planted:* "large" must mean large *imbalance* (§5.3). *Transfer:* the scaling
version is general (Chizat, Oyallon, Bach; §7); the imbalance version differs in
piecewise-linear networks (Kunin et al.); lazy-then-rich has been proposed as a mechanism for
grokking (Kumar et al.).

<a id="td1-ex6-1"></a>

**Exercise 6.1 — How separated must singular values be for separate stages?** Two balanced modes with $s_1>s_2$ start at the same small $a_0$. Compare the gap between their half-times with the width of a single transition. When are they visibly separate steps? Evaluate for $s_1=1$, $s_2=0.9$, $a_0=10^{-3}$, and for $s_1=1$, $s_2=0.5$.

<details><summary>Hint</summary>

Use $t_{1/2}=(\tau/2s)\ln(s/a_0-1)$ and the rise time of a balanced mode: by Exercise 1.4.2, going from $\epsilon s$ to $(1-\epsilon)s$ takes $(\tau/s)\ln\frac{1-\epsilon}{\epsilon}$, independent of $a_0$ (take $\epsilon=0.1$ for 10%–90%).
</details>
"""
    ),
    answer(
        r"""
Half-times $t_\alpha\approx\frac{\tau}{2s_\alpha}\ln\frac{s_\alpha}{a_0}$. Ignoring the slow variation of $\ln s_\alpha$, the gap is $t_2-t_1\approx\frac{\tau}{2}\ln\frac1{a_0}\big(\frac1{s_2}-\frac1{s_1}\big)\approx\frac{\tau}{2}\ln\frac1{a_0}\,\frac{\delta s}{s^2}$ with $\delta s=s_1-s_2$. The transition from 10% to 90% takes $\frac{\tau}{s}\ln9\approx2.2\,\tau/s$ (Exercise 1.4.2 with $\epsilon=0.1$). Separate steps need gap $\gg$ width, i.e. $\frac{\delta s}{s}\gg\frac{2\ln9}{\ln(1/a_0)}\approx\frac{4.4}{\ln(1/a_0)}$. Smaller initialization helps, but only logarithmically.

For $s_1=1$, $s_2=0.9$, $a_0=10^{-3}$: $t_1=\tfrac12\ln999\approx3.45\,\tau$ and $t_2=\tfrac1{1.8}\ln899\approx3.78\,\tau$. The gap $0.33\,\tau$ is much smaller than a transition width of about $2.2\,\tau$: one blurred step. For $s_2=0.5$: $t_2=\ln499\approx6.2\,\tau$, a gap of $2.8\,\tau$, comparable to the widths (about 2.2 and 4.4): two partly overlapping steps.
"""
    ),
    markdown(
        r"""
<a id="td1-ex6-2"></a>

**Exercise 6.2 (code) — Plant a hierarchy, then find it.** Put 8 items at the leaves of a
binary tree of depth 3 and generate many $\pm1$ features by branching diffusion: the root is
$\pm1$, and each child copies its parent but flips sign with probability $\epsilon=0.15$. With
one-hot (white) inputs, $\Sigma\propto Y$, the feature-by-item matrix. Show that its right
singular vectors are the tree's splits, with singular values decreasing level by level, and
compare a "flat" dataset of i.i.d. $\pm1$ features. Then, for $\epsilon\in\{0.05,0.15,0.3\}$,
plot the loss curve the balanced mode solutions predict ($a_0=10^{-4}$) and use Exercise 6.1
to say when stages are visible. What did the data put in that the dynamics "discovered"?
"""
    ),
    code_answer(
        r"""
rng6 = np.random.default_rng(0)
def tree_features(n_feat, depth=3, eps=0.15):
    # TODO: return an (n_feat x 2**depth) matrix of +-1 features generated by branching diffusion
    raise NotImplementedError

# TODO: SVD of Y_tree / sqrt(n_feat) and of a flat random +-1 matrix; plot the right singular vectors
#       (as an image, items on the x-axis) and the singular values.
""",
        r"""
rng6 = np.random.default_rng(0)
def tree_features(n_feat, depth=3, eps=0.15):
    vals = rng6.choice([-1.0, 1.0], size=(n_feat, 1))             # root
    for _ in range(depth):                                         # each node has two children
        children = np.repeat(vals, 2, axis=1)
        flips = rng6.random(children.shape) < eps
        vals = np.where(flips, -children, children)
    return vals                                                    # n_feat x 2**depth leaves

n_feat = 4000
Y_tree = tree_features(n_feat)
Y_flat = rng6.choice([-1.0, 1.0], size=(n_feat, 8))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, Y, name in [(axs[0], Y_tree, "tree"), (axs[1], Y_flat, "flat")]:
    _, sv, Vt = np.linalg.svd(Y / np.sqrt(n_feat), full_matrices=False)
    Vt = Vt * np.sign(Vt[:, :1] + 1e-12)                           # fix signs for display
    ax.imshow(Vt, cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax.set_title(f"{name}: right singular vectors (rows)"); ax.set_xlabel("item"); ax.set_ylabel("mode")
    axs[2].plot(range(1, 9), sv, "o-", label=name)
    print(f"{name:5s} singular values: {np.round(sv, 3)}")
axs[2].set_title("singular values"); axs[2].set_xlabel("mode"); axs[2].legend()
plt.tight_layout(); plt.show()

# Close the loop: the loss curve the mode ODEs predict for each level of hierarchy (a0 = 1e-4)
t = np.linspace(0, 12, 600)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for eps in [0.05, 0.15, 0.3]:
    sv = np.linalg.svd(tree_features(n_feat, eps=eps) / np.sqrt(n_feat), compute_uv=False)
    loss = 0.5 * sum((s_ - logistic(t, s_, 1e-4))**2 for s_ in sv)
    ax.plot(t, loss / loss[0], label=f"eps = {eps}: s = {np.round(sv[:4], 2)}...")
ax.set_xlabel("t / tau"); ax.set_ylabel("loss / initial loss"); ax.legend(fontsize=7)
ax.set_title("visible stages need strongly hierarchical data (small eps)")
plt.tight_layout(); plt.show()
""",
    ),
    answer(
        r"""
**Tree.** The first right singular vector is (nearly) constant over items: the root-level feature mean. The second splits the items 4 | 4, the next two split each half 2 | 2 (a degenerate pair, so the solver may return any rotation of them), and the last four split sibling pairs. The singular values come in groups of sizes 1, 1, 2, 4 and decrease level by level, because a feature shared by more items, after more steps of correlated diffusion, contributes more variance in that direction. So learning in order of $s$ means learning in order of tree level: the data put in the order, and the dynamics turn it into a time course. Whether it shows *separate stages* depends on the gaps. At $\epsilon=0.15$ the singular values (about 1.55, 1.24, 1.0, 0.72) are too close by Exercise 6.1's criterion, and the loss falls in one broad drop. At $\epsilon=0.05$ the root mode is far above the rest (about 2.3 against 1.0, 0.7 and 0.44) and its stage stands out, followed by a slower tail. At $\epsilon=0.3$ the spectrum is nearly flat and there is a single drop. Visible stages need data whose hierarchy is strong, which is itself part of what is planted.

**Flat.** The singular values are all close to 1 (a Marchenko–Pastur cloud), and the singular vectors are arbitrary mixtures of items with no hierarchical meaning. The gaps are far too small for separate stages, so the transitions overlap into one smooth drop.

So "progressive differentiation" is a consequence of the dynamics *given* a spectrum in which coarse splits dominate. The dynamics are not evidence that conceptual data has such a spectrum.
"""
    ),
    markdown(
        r"""
<a id="td1-ex6-3"></a>

**Exercise 6.3 — The kernel at an orthogonal initialization.** Let every layer of a depth-$N$ square linear network be orthogonal.

1. Show that the network is balanced, that $W$ is orthogonal, and that the end-to-end kernel at initialization is $K_0[E]=N\,E$. Contrast this with a small initialization, and say what the kernel has to do during training.
2. For $N=2$ and balanced weights, show that each singular value $\sigma_i$ of $W$ obeys $\tau\dot\sigma_i=2\sigma_i\,(u_i^\top Ev_i)$, where $u_i,v_i$ are its singular vectors (assume they are distinct). Deduce that $\det W$ cannot change sign. What happens if $\det W(0)$ and $\det\Sigma$ have opposite signs?

<details><summary>Hint</summary>

For a simple singular value, first-order perturbation theory gives $\dot\sigma_i=u_i^\top\dot Wv_i$. Use §4.2's balanced flow. For the determinant, which avoids assuming distinct singular values, use $\frac{d}{dt}\ln|\det W|=\operatorname{tr}(W^{-1}\dot W)$ and the polar decomposition $W=PO$ with $P=(WW^\top)^{1/2}$.
</details>
"""
    ),
    answer(
        r"""
**1.** $W_{j+1}^\top W_{j+1}=I=W_jW_j^\top$, so every adjacent pair is balanced, and $W$, a product of orthogonal matrices, is orthogonal. Then $WW^\top=W^\top W=I$, and every term of the Arora–Cohen–Hazan sum $\sum_j(WW^\top)^{(N-j)/N}E(W^\top W)^{(j-1)/N}$ is $E$, so $K_0[E]=NE$. (For $N=2$ directly: $W_2W_2^\top E+EW_1^\top W_1=2E$.) A small initialization of per-layer scale $w_0$ has kernel $O(Nw_0^{2N-2})$ on every mode, nearly zero, hence the plateaus. The orthogonal start's isotropic kernel moves every direction at once. It is not frozen, though: if training converges, $W$ goes from an orthogonal matrix to $\Sigma$, and the balanced kernel changes from $N\cdot\mathrm{Id}$ to a function of $\Sigma$, by $O(1)$ unless $\Sigma$ is itself close to orthogonal. Orthogonal initialization avoids the small-kernel plateau; it does not make training lazy.

**2.** For simple singular values, $\dot\sigma_i=u_i^\top\dot Wv_i$. With the balanced kernel of §4.2, $\tau\dot W=(WW^\top)^{1/2}E+E(W^\top W)^{1/2}$, and $(WW^\top)^{1/2}u_i=\sigma_iu_i$, $(W^\top W)^{1/2}v_i=\sigma_iv_i$, so $\tau\dot\sigma_i=2\sigma_i\,u_i^\top Ev_i$. Since $E$ stays bounded (the loss decreases), $|\tau\dot\sigma_i|\le C\sigma_i$ and $\sigma_i(t)\ge\sigma_i(0)e^{-Ct/\tau}>0$. This needs distinct singular values, which fails at an orthogonal start, so argue with the determinant instead. While $W$ is invertible, write $W=PO$ with $O$ orthogonal; then $(W^\top W)^{1/2}=O^\top PO$ and $\tau\frac{d}{dt}\ln|\det W|=\operatorname{tr}\big(W^{-1}(PE+EO^\top PO)\big)=2\operatorname{tr}(O^\top E)$, which is bounded by $2\sqrt n\,\|E\|_F$. So $\ln|\det W|$ cannot reach $-\infty$ in finite time: $\det W$ never vanishes, and its sign is fixed. If it starts opposite to $\det\Sigma$, the flow can never reach $\Sigma$. Instead one singular value decays toward zero and $W$ approaches a saddle of §3 (in the $1\times1$ example $W_1=1$, $W_2=-1$, $s=1$, both weights shrink to zero, and gradient descent keeps $W_2=-W_1$ exactly). In generic matrix runs, finite steps eventually generate enough imbalance to escape, after a long stall.
"""
    ),
    markdown(
        r"""
The cell checks Exercise 6.3.2: four random orthogonal initializations, two with
$\det W(0)$ of the same sign as $\det\Sigma$ and two opposite (one row of $W_1$ flipped).
"""
    ),
    code(
        r"""
r = np.random.default_rng(0); n = 4
Sig4, U4, V4 = make_target([3.0, 2.0, 1.0, 0.5], n=4, seed=11)
for want in [+1, +1, -1, -1]:
    W1 = np.linalg.qr(r.normal(size=(n, n)))[0]; W2 = np.linalg.qr(r.normal(size=(n, n)))[0]
    if np.sign(np.linalg.det(W2 @ W1) * np.linalg.det(Sig4)) != want:
        W1[0] *= -1                                    # still orthogonal and balanced
    ts, hist = gd([W1, W2], Sig4, 0.002, 7500, every=7500)
    Wf = product(hist[-1])
    print(f"det sign {'matches' if want > 0 else 'opposite'}: loss at t = 15 tau = "
          f"{0.5 * np.sum((Sig4 - Wf)**2):.4f}, singular values of W = {np.round(np.linalg.svd(Wf)[1], 4)}")
print("loss of the saddle that drops the weakest mode: 0.5 * 0.5**2 =", 0.5 * 0.25)
"""
    ),
    # ================================================================== §7
    markdown(
        r"""
<a id="td1-s7"></a>

## 7. Going further

**The analytic escape hatch.** For a day of exactly solvable theory instead, the Physics of
Deep Learning day builds the kernel story from scratch:
[Notebook 1, *Fixed features and kernel learning*]("""
        + PHYS
        + r"""/01_fixed_features_kernel_learning_solutions.ipynb) (gradient flow as a spectral filter),
[Notebook 2, *Why wide neural networks become kernel learners*]("""
        + PHYS
        + r"""/02_ntk_limit_kernel_learners_solutions.ipynb) (the NTK and NNGP limits), and
[Notebook 3, *Beyond fixed kernels*]("""
        + PHYS
        + r"""/03_beyond_ntk_feature_learning_and_cumulants_solutions.ipynb) (moving kernels, cumulants, and an exactly solvable deep linear network at finite width).

**In NB2.** The mode picture returns in [what is learned first](td2_7_learned_first.html),
the kernel $K_t$ in [curvature](td2_6_curvature.html) (as the Gauss–Newton matrix), and
muP in [optimizer geometry](td2_4_optimizer_geometry.html).
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "td1_linear_networks", "td1_linear_networks"))
