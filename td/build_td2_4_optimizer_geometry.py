#!/usr/bin/env python3
"""Build NB2 §4, "Optimizers as steepest descent under a norm" (Training Dynamics day).

Writes td2_4_optimizer_geometry.html (the reading page) and .ipynb (the runnable
notebook) from one source.

Shape (readings first, 2026-10-14): a short pitch, a one-line route and a folded box with
the SGD-to-AdamW update rules; then one `## Readings` section (anchor td24-readings) with
every paper of the notebook, grouped by section; then the derivations and walkthrough,
sections 4.0–4.6 (anchors td24-0 … td24-6). Each section is a few sentences, a one-line
pointer to its paper(s), and a short core-idea paragraph, followed (from the
td24-N-walkthrough anchor) by the derivations, whose exercises and code demos are
bold-labelled paragraphs (Exercise 4.N.X), not headings.

Closed forms are checked by checks/td2_4_optimizer_geometry_math.py; the interactive
figure's math by assets/interactive/checks/steepest_descent_balls_check.py. Review log:
../reviews/td/td2_4_optimizer_geometry.md.
"""
from nbkit_td import answer, code, code_answer, figure, markdown, reading, run

STEM = "td2_4_optimizer_geometry"

BN_URL = "https://arxiv.org/abs/2409.20325"
MUON_URL = "https://kellerjordan.github.io/posts/muon/"
YSB_URL = "https://arxiv.org/abs/2310.17813"

CELLS = [
    # ================================================================== pitch
    markdown(
        r"""
# NB2 §4 — Optimizers as steepest descent under a norm

**Training Dynamics, Iliad Intensive 2026**

To first order a step $\Delta$ changes the loss by $\langle g,\Delta\rangle$, with
$g=\nabla L$. That has no minimum until you bound the step's size, and the norm you bound
it in fixes the direction: $\ell_2$ gives gradient descent, $\ell_\infty$ gives sign descent
(Adam with its averages off), the Fisher norm gives natural gradient, and the spectral norm
gives Muon's orthogonalized update. Asking which norm lets step sizes transfer across width
gives muP.

**Route.** The [Readings](#td24-readings) come first: every paper of the notebook, grouped by
section. The sections after them are the derivations and walkthrough (exercises with worked
solutions, and code), for whoever wants them.

- [4.0 A parable](#td24-0): between disjoint distributions, KL and JS give no gradient and Wasserstein does.
- [4.1 Steepest descent under a norm](#td24-1) (core): the norm-to-optimizer dictionary and the interactive unit balls.
- [4.2 Natural gradient](#td24-2): invariance to how parameters are written, and why raw sharpness lacks it.
- [4.3 Adam as smoothed sign descent](#td24-3): when the sign picture holds, and AdamW's implicit box.
- [4.4 Newton–Schulz](#td24-4): how Muon orthogonalizes without an SVD.
- [4.5 Width](#td24-5): muP's learning rates from the spectral condition.
- [4.6 The modular norm](#td24-6): one norm for the whole network, and depth.

<a id="td24-orientation"></a>

<details><summary>Update rules from SGD to AdamW, if you need them</summary>

$\theta$ are the parameters, $g$ the minibatch gradient, $\eta$ the learning rate, $\lambda$
the weight decay, $\beta,\beta_1,\beta_2\in[0,1)$ EMA (exponential moving average) rates.
Operations are per coordinate ($g^2=g\odot g$), $t$ counts steps from 1, and $m,v$ start at 0.

| Optimizer | Update | Where it sits here |
|---|---|---|
| SGD | $\theta\leftarrow\theta-\eta g$ | steepest descent in $\ell_2$, penalty form (§4.1) |
| Heavy ball | $m\leftarrow\beta m+g,\quad \theta\leftarrow\theta-\eta m$ | same geometry as SGD; momentum filters the gradient sequence |
| RMSProp | $v\leftarrow\beta_2v+(1-\beta_2)g^2,\quad \theta\leftarrow\theta-\eta\,g/(\sqrt v+\epsilon)$ | diagonal preconditioner; sign descent ($\ell_\infty$) at $\beta_2=\epsilon=0$ |
| Adam | $m\leftarrow\beta_1m+(1-\beta_1)g,\ \ v\leftarrow\beta_2v+(1-\beta_2)g^2,\ \ \hat m=\dfrac{m}{1-\beta_1^t},\ \ \hat v=\dfrac{v}{1-\beta_2^t},\ \ \theta\leftarrow\theta-\eta\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}$ | smoothed sign descent (§4.3) |
| Adam + L2 | Adam with $g\leftarrow g+\lambda\theta$ | the decay is divided by $\sqrt{\hat v}$ too |
| AdamW | $\theta\leftarrow\theta-\eta\big(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta\big)$ | decay outside the preconditioner; an implicit box $\lvert\theta_i\rvert\lesssim1/\lambda$ (Exercise 4.3.E) |

Momentum filters the gradient sequence (its stability is in [NB2 §1](td2_1_flows.html#td21-s5));
the norm then decides what step a filtered gradient becomes: Adam normalizes its momentum per
coordinate, Muon orthogonalizes it. Adam's bias correction undoes the zero start of $m$ and $v$
([Kingma and Ba 2015](https://arxiv.org/abs/1412.6980)); defaults are $\beta_1=0.9$,
$\beta_2=0.999$, $\epsilon=10^{-8}$. AdamW ([Loshchilov and Hutter 2019](https://arxiv.org/abs/1711.05101))
decouples the decay so that weights with a large gradient history are still shrunk; its role
through normalization layers is [NB2 §5](td2_5_normalization_weight_decay.html#s52).
</details>
"""
    ),
    code(
        r"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
"""
    ),
    # ================================================================== readings
    markdown(
        r"""
<a id="td24-readings"></a>

## Readings

**A parable: distances between distributions ([§4.0](#td24-0)).**
"""
        + reading(
            "KL divergence, geometric naturalness, and optimal transport",
            "https://charlesr-w.github.io/crw-blog/KL-Divergence-Geometric-Naturalness-and-Optimal-Transport/",
            "Charles Renshaw-Whitman, blog post (2025)",
            "§4.0's point-mass example, the case for metric-aware distances between distributions, "
            "and WGAN's dual workaround.",
            importance=4,
            length=2,
        )
        + r"""
*Also:* [WGAN](https://arxiv.org/abs/1701.07875) (its Example 1: two parallel segments);
[Arjovsky and Bottou](https://arxiv.org/abs/1701.04862) (why a GAN generator's gradient vanishes).

**Steepest descent under a norm ([§4.1](#td24-1)).** The notebook's lead paper.
"""
        + reading(
            "Old optimizer, new norm: an anthology",
            BN_URL,
            "Jeremy Bernstein and Laker Newhouse (OPT 2024 workshop)",
            "Short and readable: with their moving averages switched off, Adam, Shampoo and Prodigy are "
            "steepest descent under different norms; the source of §4.1's dictionary.",
            importance=5,
            length=2,
        )
        + r"""
**Natural gradient ([§4.2](#td24-2)).**
"""
        + reading(
            "New insights and perspectives on the natural gradient method",
            "https://arxiv.org/abs/1412.1193",
            "James Martens (JMLR 2020)",
            "The careful version of §4.2: the relation to the generalized Gauss–Newton matrix, "
            "approximate (not exact) invariance for finite steps, and why the Hessian lacks it.",
            importance=2,
            length=4,
        )
        + r"""
*Also:* [Amari 1998](https://doi.org/10.1162/089976698300017746) (the original);
[Kunstner, Balles and Hennig](https://arxiv.org/abs/1905.12558) (the empirical Fisher);
[K-FAC](https://arxiv.org/abs/1503.05671); [Dinh et al.](https://arxiv.org/abs/1703.04933)
(sharp minima made flat by rescaling).

**Adam as smoothed sign descent ([§4.3](#td24-3)).**
"""
        + reading(
            "Noise is not the main factor behind the gap between SGD and Adam on transformers, but sign descent might be",
            "https://arxiv.org/abs/2304.13960",
            "Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, Mark Schmidt (ICLR 2023)",
            "Growing the batch to the full dataset widens Adam's lead over SGD, and sign descent with "
            "momentum behaves much like Adam at large batch.",
            importance=3,
            length=2,
        )
        + r"""
*Also:* [Kunstner et al. 2024](https://arxiv.org/abs/2402.19449) (heavy-tailed class
imbalance); [Lion](https://arxiv.org/abs/2302.06675) (sign descent with momentum, found by program search);
[Balles and Hennig](https://arxiv.org/abs/1705.07774) (Adam as sign plus variance adaptation).

**Newton–Schulz and Muon ([§4.4](#td24-4)).**
"""
        + reading(
            "Muon: an optimizer for hidden layers in neural networks",
            MUON_URL,
            "Keller Jordan et al. (blog, 2024)",
            "Muon's definition, its Newton–Schulz coefficients and how they were tuned, and the "
            "NanoGPT-speedrun evidence (§4.4).",
            importance=4,
            length=2,
        )
        + r"""
**Width and muP ([§4.5](#td24-5)).**
"""
        + reading(
            "A spectral condition for feature learning",
            YSB_URL,
            "Greg Yang, James Simon, Jeremy Bernstein (2023)",
            "An elementary derivation of muP from one condition on the spectral norms of weights and "
            "updates (§4.5); short enough to read in full.",
            importance=4,
            length=2,
        )
        + reading(
            "Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer",
            "https://arxiv.org/abs/2203.03466",
            "Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, "
            "Jakub Pachocki, Weizhu Chen, Jianfeng Gao (NeurIPS 2021)",
            "muTransfer, and in Table 3 the muP scalings per layer type for SGD and Adam.",
            importance=2,
            length=3,
        )
        + r"""
*Also:* [Tensor Programs IV](https://arxiv.org/abs/2011.14522) (muP's original derivation);
[Moonlight](https://arxiv.org/abs/2502.16982) (a different Muon scaling convention).

**The modular norm ([§4.6](#td24-6)).**
"""
        + reading(
            "Scalable optimization in the modular norm",
            "https://arxiv.org/abs/2405.14813",
            "Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, Jeremy Bernstein (NeurIPS 2024)",
            "Modules with masses and sensitivities, the composition rules, smoothness in the modular "
            "norm, and learning-rate transfer across width and depth; comes with the Modula package.",
            importance=3,
            length=3,
        )
        + r"""
*Also:* [Modular duality](https://arxiv.org/abs/2410.21265) (the gradient-to-update map for
the whole network, with Newton–Schulz updates for linear and convolutional layers).
"""
    ),
    # ================================================================== 4.0
    markdown(
        r"""
<a id="td24-0"></a>

## 4.0 A parable: choose a metric that respects the space

Between two distributions with disjoint supports, KL is infinite and Jensen–Shannon is
constant, so neither gives gradient descent anything to follow. The Wasserstein distance
does, because it measures how far mass has to move.

**Paper:** Renshaw-Whitman, *KL divergence, geometric naturalness, and optimal transport*
(see [Readings](#td24-readings)).

**The analogy, labelled as one.** Gradient descent measures a step by the Euclidean norm of
the raw parameters, as if every coordinate were an interchangeable label. The
parameter-space analogue of "use Wasserstein" is to measure an update by how much it changes
a layer's outputs: the spectral and modular norms of §4.1, §4.5 and §4.6. Natural gradient
(§4.2) is a different move. It removes dependence on how the parameters are written, but it
measures steps by KL, and so inherits KL's blindness to output geometry (Exercise 4.2.C).

<a id="td24-0-walkthrough"></a>

**Derivations and walkthrough.** Early in GAN training the generator's distribution $P_\theta$ typically lives on a
low-dimensional set that misses the data. KL and JS compare distributions point by point, so
moving mass a tiny distance onto empty ground costs as much as moving it anywhere; the
Wasserstein-1 distance $W_1$ (earth mover's distance) asks how much mass moves and how far.
WGAN estimates it through the Kantorovich–Rubinstein dual,
$W_1(P,Q)=\sup_{\|f\|_{\rm Lip}\le1}\mathbb E_Pf-\mathbb E_Qf$, with a critic network as $f$.

**Exercise 4.0.A — Point masses.** Take $P_0=\delta_0$ and $P_\theta=\delta_\theta$ on
$\mathbb R$. Here $W_1(P,Q)=\min\mathbb E|X-Y|$ over couplings (joint laws of $(X,Y)$ with
marginals $P$ and $Q$).

1. Compute $\mathrm{KL}(P_0\|P_\theta)$, $\mathrm{JS}(P_0,P_\theta)=\tfrac12\mathrm{KL}(P_0\|M)+\tfrac12\mathrm{KL}(P_\theta\|M)$ with $M=\tfrac12(P_0+P_\theta)$, and $W_1(P_0,P_\theta)$, for every $\theta$.
2. Which of the three could gradient descent use to bring $\theta$ to 0?

<details><summary>Hint</summary>

For KL, ask whether $P_0$ puts mass where $P_\theta$ puts none. For JS, both
distributions are absolutely continuous with respect to $M$, with density 2 on their own
atom. For $W_1$, there is only one coupling of two point masses.
</details>
"""
    ),
    answer(
        r"""
**1.** *KL.* For $\theta\ne0$, $P_0$ puts mass 1 on $\{0\}$, where $P_\theta$ puts none, so $P_0$ is not absolutely continuous with respect to $P_\theta$ and $\mathrm{KL}(P_0\|P_\theta)=+\infty$. At $\theta=0$ it is 0.

*JS.* For $\theta\ne0$, $M$ puts mass $\tfrac12$ on each atom, so $dP_0/dM=2$ on $\{0\}$ and $\mathrm{KL}(P_0\|M)=\log2$. The same holds for $P_\theta$, so $\mathrm{JS}=\log2$, the largest value JS can take. At $\theta=0$ it is 0.

*$W_1$.* The only coupling of $\delta_0$ and $\delta_\theta$ is $\delta_{(0,\theta)}$, which moves unit mass a distance $|\theta|$. So $W_1=|\theta|$.

**2.** JS is constant on $\theta\ne0$, so its derivative there is zero and gradient descent never moves. KL is $+\infty$ for every $\theta\ne0$, so there is no finite objective and no gradient. $\frac{d}{d\theta}W_1=\mathrm{sign}(\theta)$ for $\theta\ne0$: every step moves $\theta$ a distance $\eta$ toward 0, ending in an oscillation of size at most $\eta$ around 0, which a step size $\eta_t\to0$ with $\sum_t\eta_t=\infty$ removes. Only $W_1$ is usable.
"""
    ),
    # ================================================================== 4.1
    markdown(
        r"""
<a id="td24-1"></a>

## 4.1 Steepest descent under a norm

The linear change $\langle g,\Delta\rangle$ has a steepest direction only once you say how a
step's size is measured. Each norm picks one, and the common optimizers are the answers for
$\ell_2$, $\ell_\infty$, a Mahalanobis norm and the spectral norm.

**Paper:** Bernstein and Newhouse, *Old optimizer, new norm* (see [Readings](#td24-readings));
its Proposition 1 is the penalty formula below.

Write $\|g\|_*=\max_{\|d\|\le1}\langle g,d\rangle$ for the **dual norm** and $d^\star(g)$ for a
unit-ball point attaining it. Minimizing $\langle g,\Delta\rangle$ over the ball
$\|\Delta\|\le\eta$ gives the **trust-region** step $-\eta\,d^\star(g)$, of fixed length $\eta$.
Minimizing $\langle g,\Delta\rangle+\|\Delta\|^2/(2\eta)$ gives the **penalty** step
$-\eta\,\|g\|_*\,d^\star(g)$, which shrinks as the gradient does. Same direction, different
lengths:

| Norm on $\Delta$ | Dual norm $\lVert g\rVert_*$ | Trust-region step | Penalty step | Known as |
|---|---|---|---|---|
| $\ell_2$ | $\lVert g\rVert_2$ | $-\eta\,g/\lVert g\rVert_2$ | $-\eta\,g$ | normalized GD; GD |
| $\ell_\infty$ | $\lVert g\rVert_1$ | $-\eta\,\mathrm{sign}(g)$ | $-\eta\lVert g\rVert_1\mathrm{sign}(g)$ | sign descent (Adam, RMSProp with EMAs off) |
| $\ell_1$ | $\lVert g\rVert_\infty$ | $-\eta\,\mathrm{sign}(g_i)e_i$, $i=\arg\max_j\lvert g_j\rvert$ | $-\eta\lVert g\rVert_\infty\mathrm{sign}(g_i)e_i$ | greedy coordinate descent |
| $\sqrt{\Delta^\top M\Delta}$ | $\sqrt{g^\top M^{-1}g}$ | $-\eta\,M^{-1}g/\sqrt{g^\top M^{-1}g}$ | $-\eta\,M^{-1}g$ | preconditioned GD; natural gradient for $M=F$ |
| spectral $\lVert\Delta\rVert_2$ (matrix) | $\lVert G\rVert_{\rm nuc}$ | $-\eta\,UV^\top$ | $-\eta\lVert G\rVert_{\rm nuc}UV^\top$ | Shampoo without accumulation; Muon (approximately, applied to momentum) |

Here $M\succ0$ is a fixed matrix and $F$ the Fisher information (§4.2). In the last row the
step is a matrix, $G=U\Sigma V^\top$ is a reduced SVD of the gradient, $\|\cdot\|_2$ is the
largest singular value and $\|\cdot\|_{\rm nuc}$ the sum of the singular values.
"""
        + "\n\n"
        + figure(
            "assets/interactive/steepest_descent_balls.html",
            "Drag a 2D gradient and watch the unit balls of $\\ell_2$, $\\ell_\\infty$, $\\ell_1$ and a "
            "Mahalanobis (Fisher-like) norm pick out their steepest-descent steps, in trust-region "
            "or penalty form.",
        )
        + r"""

<a id="td24-1-walkthrough"></a>

**Derivations and walkthrough.** Notation as above: $g\ne0$ the gradient, $\|\cdot\|$ a norm on steps, $\|g\|_*$ its dual,
$d^\star(g)$ a maximizer. Four remarks first.

1. **Plain GD is the penalty form in $\ell_2$.** Adam and Muon behave more like trust-region methods, so near a minimum, where the gradient shrinks but their steps do not, they need a decaying $\eta$ (Exercise 4.1.D).
2. **Corners make the direction non-unique.** For $\ell_\infty$, $\ell_1$ and spectral balls the maximizer can be a whole face: $\mathrm{sign}(0)$ is anything in $[-1,1]$; ties in $|g_i|$ for $\ell_1$; zero singular values for the spectral norm.
3. **Layers compose by a max.** Measure the whole step by $\max_l\|\Delta_l\|_{(l)}$. The dual norm is $\sum_l\|g_l\|_{(l),*}$, and the trust-region step is every layer's own steepest step with the same radius $\eta$. Bernstein and Newhouse read Adam this way: $\ell_\infty$ on all the weights is the max over layers of each matrix's $\ell_1\to\ell_\infty$ operator norm, its largest entry $\max_{ij}|W_{ij}|$. How the radii should differ between layers is §4.5–§4.6.
4. **The dictionary is not a justification.** Every optimizer is steepest descent under *some* norm. A norm earns its place by what it buys: invariance to the parametrization (§4.2), or step sizes that transfer across width (§4.5–§4.6).

**Exercise 4.1.A — Both forms.** For an arbitrary norm:

1. Show that $\min_{\|\Delta\|\le\eta}\langle g,\Delta\rangle=-\eta\|g\|_*$, attained at $\Delta=-\eta d^\star(g)$.
2. Show that $\Delta_{\rm pen}=\arg\min_\Delta\langle g,\Delta\rangle+\|\Delta\|^2/(2\eta)$ is $-\eta\|g\|_*d^\star(g)$, with optimal value $-\tfrac\eta2\|g\|_*^2$.

<details><summary>Hint</summary>

For 1, norm balls are symmetric, $d\in B\Leftrightarrow -d\in B$. For 2, write
$\Delta=c\,u$ with $c\ge0$ and $\|u\|=1$, minimize over $u$ at fixed $c$, then over $c$.
</details>
"""
    ),
    answer(
        r"""
**1.** For $\|\Delta\|\le\eta$, write $\Delta=-\eta d$ with $\|d\|\le1$; the ball is symmetric, so $d$ ranges over the whole unit ball. Then $\langle g,\Delta\rangle=-\eta\langle g,d\rangle\ge-\eta\max_{\|d\|\le1}\langle g,d\rangle=-\eta\|g\|_*$, with equality at $d=d^\star(g)$. (If $g=0$, every point of the ball is a minimizer.)

**2.** Write $\Delta=c\,u$ with $c\ge0$, $\|u\|=1$. The objective is $c\langle g,u\rangle+c^2/(2\eta)$. At fixed $c$ the first term is smallest when $u=-d^\star(g)$; for $g\ne0$ we can take $\|d^\star\|=1$, since a nonzero linear function on a ball is maximized on the boundary. This gives $-c\|g\|_*+c^2/(2\eta)$, minimized at $c=\eta\|g\|_*$ with value $-\tfrac\eta2\|g\|_*^2$. So $\Delta_{\rm pen}=-\eta\|g\|_*d^\star(g)$: the trust-region direction, with length set by the dual norm of the gradient.
"""
    ),
    markdown(
        r"""
**Exercise 4.1.B — Three vector norms, and Newton.** Fill in the dictionary's vector rows:
for each norm find $\|g\|_*$ and $d^\star$.

1. $\ell_\infty$: show $\|g\|_*=\|g\|_1$ and $d^\star=\mathrm{sign}(g)$.
2. $\ell_1$: show $\|g\|_*=\|g\|_\infty$ and $d^\star=\mathrm{sign}(g_i)e_i$ with $i=\arg\max_j|g_j|$.
3. Mahalanobis, $\|\Delta\|_M=\sqrt{\Delta^\top M\Delta}$ with $M\succ0$: show $\|g\|_*=\sqrt{g^\top M^{-1}g}$ and $d^\star=M^{-1}g/\|g\|_*$, so the penalty step is $-\eta M^{-1}g$.
4. Which $M$ and $\eta$ make the penalty step Newton's step $-H^{-1}g$? On a quadratic $L(\theta)=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$ with $H\succ0$, where does it land? Why is Newton not usually called steepest descent?

<details><summary>Hint</summary>

For 3, write $\langle g,d\rangle=\langle M^{-1/2}g,\,M^{1/2}d\rangle$ and use Cauchy–Schwarz.
</details>
"""
    ),
    answer(
        r"""
**1.** $\langle g,d\rangle=\sum_ig_id_i\le\sum_i|g_i|\,|d_i|\le\|d\|_\infty\|g\|_1$, with equality when every $d_i=\mathrm{sign}(g_i)$. Coordinates with $g_i=0$ can take any value in $[-1,1]$ (remark 2). The trust-region step $-\eta\,\mathrm{sign}(g)$ moves *every* coordinate by exactly $\eta$, however small its gradient.

**2.** $\langle g,d\rangle\le\sum_i|g_i||d_i|\le\max_j|g_j|\sum_i|d_i|\le\|g\|_\infty$ for $\|d\|_1\le1$, with equality when all the mass of $d$ sits on a largest coordinate with the right sign: greedy coordinate descent. Ties give a whole face of maximizers.

**3.** $\langle g,d\rangle=\langle M^{-1/2}g,M^{1/2}d\rangle\le\|M^{-1/2}g\|_2\,\|M^{1/2}d\|_2=\sqrt{g^\top M^{-1}g}\;\|d\|_M$, with equality when $M^{1/2}d\propto M^{-1/2}g$, i.e. $d\propto M^{-1}g$. Normalizing, $d^\star=M^{-1}g/\sqrt{g^\top M^{-1}g}$, and the penalty step is $-\eta\|g\|_*d^\star=-\eta M^{-1}g$: preconditioned gradient descent. With $M=F$, the Fisher information, it is Amari's natural gradient. Since $\mathrm{KL}(p_\theta\|p_{\theta+\delta})=\tfrac12\delta^\top F\delta+O(\|\delta\|^3)$, $\|\Delta\|_F^2/2$ is, to second order, the KL divergence between the model before and after the step: natural gradient measures a step by how much the predictive distribution changes.

**4.** $M=H$, $\eta=1$. On the quadratic, $g=H(\theta-\theta^*)$, so the step lands exactly on $\theta^*$. Newton is steepest descent in the norm of the loss's own curvature. It is not usually described that way because away from a minimum $H$ need not be positive definite, so $\sqrt{\Delta^\top H\Delta}$ need not be a norm, and because this "norm" changes with the loss. For exponential-family likelihoods with the canonical link, the Fisher equals the generalized Gauss–Newton matrix, a positive semidefinite approximation to $H$ ([Martens](https://arxiv.org/abs/1412.1193)); this is one reason natural gradient sits between gradient descent and Newton.
"""
    ),
    markdown(
        r"""
**Exercise 4.1.C — The spectral norm, $UV^\top$, and Shampoo.** For a weight matrix, measure
a step by how much it can stretch a vector. Notation: $G\in\mathbb R^{m\times n}$ with reduced
SVD $G=\sum_{i=1}^r\sigma_iu_iv_i^\top=U\Sigma V^\top$, $r=\mathrm{rank}\,G$;
$\langle A,B\rangle=\mathrm{tr}(A^\top B)$; $\|D\|_2=\sigma_{\max}(D)$; $\|G\|_{\rm nuc}=\sum_i\sigma_i$.

1. Show $\langle G,D\rangle\le\|D\|_2\,\|G\|_{\rm nuc}$ for every $D$, and that $D=UV^\top$ attains it with $\|UV^\top\|_2=1$. Conclude the last row of the table.
2. Take $m\le n$ and $G$ of full rank $m$. Show that $UV^\top$ is also the nearest *semi-orthogonal* matrix to $G$ in Frobenius norm (minimizing $\|O-G\|_F$ over $OO^\top=I_m$). Muon's post defines its target $\mathrm{Ortho}(G)$ this way.
3. When $G$ is rank-deficient, show the maximizer in 1 is not unique.
4. Shampoo steps along $L^{-1/4}GR^{-1/4}$ with accumulated $L=\sum GG^\top$, $R=\sum G^\top G$. Show that with no accumulation (only the current full-rank $G$) this is $UV^\top$.

<details><summary>Hint</summary>

For 1, expand $\langle G,D\rangle=\mathrm{tr}(G^\top D)=\sum_i\sigma_i\,u_i^\top Dv_i$ and bound
each $u_i^\top Dv_i$. For 2, expand $\|O-G\|_F^2$ and note that $\|O\|_F^2$ is the same for
every semi-orthogonal $O$. For 4, $GG^\top=U\Sigma^2U^\top$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\langle G,D\rangle=\mathrm{tr}\big(\sum_i\sigma_iv_iu_i^\top D\big)=\sum_i\sigma_i\,u_i^\top Dv_i$. Since $u_i,v_i$ are unit vectors, $|u_i^\top Dv_i|\le\|D\|_2$, so $\langle G,D\rangle\le\|D\|_2\|G\|_{\rm nuc}$. For $D=UV^\top$, $u_i^\top UV^\top v_i=1$, which gives equality, and $UV^\top$ has $r$ singular values equal to 1, so $\|UV^\top\|_2=1$. Hence the dual of the spectral norm is the nuclear norm, $d^\star(G)=UV^\top$, the trust-region step is $-\eta\,UV^\top$ and the penalty step $-\eta\|G\|_{\rm nuc}UV^\top$. Every nonzero singular value of the step is set to the same value, whatever its size in $G$.

**2.** For semi-orthogonal $O$, $\|O\|_F^2=\mathrm{tr}(OO^\top)=m$ and $\|O\|_2=1$. So $\|O-G\|_F^2=m-2\langle O,G\rangle+\|G\|_F^2$, and minimizing the distance means maximizing $\langle O,G\rangle$. By 1, $\langle O,G\rangle\le\|G\|_{\rm nuc}$, with equality at $O=UV^\top$, which is semi-orthogonal because $U$ is $m\times m$ orthogonal and $V^\top V=I_m$. It is the orthogonal factor of the polar decomposition $G=(UV^\top)(V\Sigma V^\top)$.

**3.** Let $U_\perp$, $V_\perp$ be orthonormal bases of the complements of the column spaces of $U$ and $V$, and $W$ any matrix with $\|W\|_2\le1$. Then $D=UV^\top+U_\perp WV_\perp^\top$ has $\|D\|_2=1$ (the two blocks act on orthogonal subspaces) and $\langle G,D\rangle=\|G\|_{\rm nuc}$ (because $U^\top U_\perp=0$). $W=0$ is the minimum-Frobenius-norm maximizer, and it is what the cubic Newton–Schulz iteration converges to (§4.4), since odd polynomial iterations leave zero singular values at zero.

**4.** With $G=U\Sigma V^\top$ of full rank, $(GG^\top)^{-1/4}=U\Sigma^{-1/2}U^\top$ and $(G^\top G)^{-1/4}=V\Sigma^{-1/2}V^\top$ (on their ranges), so $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}=U\Sigma^{-1/2}\,\Sigma\,\Sigma^{-1/2}V^\top=UV^\top$. Shampoo without accumulation is spectral-norm steepest descent, as Bernstein and Newhouse point out. Accumulation is where Shampoo and Muon part ways: Shampoo accumulates the gradient's statistics, Muon accumulates the gradient itself (momentum) and orthogonalizes that.
"""
    ),
    markdown(
        r"""
**Exercise 4.1.D (code) — Trust-region steps on an ill-conditioned quadratic.** Run four
optimizers on $L(\theta)=\tfrac12\theta^\top H\theta$ with $H=\mathrm{diag}(1,25)$ from
$\theta_0=(0.93,0.71)$: gradient descent (penalty form in $\ell_2$), normalized GD (trust
region in $\ell_2$), sign descent (trust region in $\ell_\infty$) and preconditioned GD with
$M=\mathrm{diag}(H)$ (penalty form, Mahalanobis). Use a constant $\eta$, then
$\eta_t=\eta_0/\sqrt{t+1}$. Which converge at constant $\eta$, which stall, and why?
"""
    ),
    code_answer(
        r"""
H = np.diag([1.0, 25.0])
theta0 = np.array([0.93, 0.71])

def update(theta, method, eta):
    g = H @ theta
    # return the new theta for method in "gd", "normalized", "sign", "precond" (M = diag(H))
    raise NotImplementedError

# Run each method for 200 steps with constant eta (try eta = 0.07 for GD, 0.7 for precond,
# 0.05 for the two trust-region methods) and with eta_t = eta0 / sqrt(t + 1); plot the loss on a log scale.
""",
        r"""
H = np.diag([1.0, 25.0])
theta0 = np.array([0.93, 0.71])
loss = lambda th: 0.5 * th @ H @ th

def update(theta, method, eta):
    g = H @ theta
    if method == "gd":
        return theta - eta * g
    if method == "normalized":
        return theta - eta * g / np.linalg.norm(g)
    if method == "sign":
        return theta - eta * np.sign(g)
    if method == "precond":
        return theta - eta * g / np.diag(H)
    raise ValueError(method)

etas = {"gd": 0.07, "normalized": 0.05, "sign": 0.05, "precond": 0.7}
labels = {"gd": "GD (ℓ2 penalty)", "normalized": "normalized GD (ℓ2 trust)",
          "sign": "sign descent (ℓ∞ trust)", "precond": "precond. GD, M = diag(H)"}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
axs[0].contour(xx, yy, 0.5 * (xx**2 + 25 * yy**2), levels=np.geomspace(1e-3, 15, 12), colors="0.8", linewidths=0.7)
for method, eta0 in etas.items():
    for ax, schedule in [(axs[1], "constant"), (axs[2], "decayed")]:
        th = theta0.copy(); losses, path = [loss(th)], [th]
        for t in range(200):
            eta = eta0 if schedule == "constant" else eta0 / np.sqrt(t + 1)
            th = update(th, method, eta); losses.append(loss(th)); path.append(th)
        ax.semilogy(losses, label=labels[method])
        if schedule == "constant":
            path = np.array(path)
            axs[0].plot(path[:40, 0], path[:40, 1], ".-", ms=3, lw=0.8, label=labels[method])
axs[0].set_title("first 40 steps, constant η"); axs[0].set_aspect("equal")
axs[1].set_title("constant η"); axs[2].set_title(r"$\eta_t=\eta_0/\sqrt{t+1}$")
for ax in axs[1:]:
    ax.set_xlabel("step"); ax.set_ylim(1e-12, 30)
axs[1].set_ylabel("loss"); axs[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""",
    ),
    answer(
        r"""
At constant $\eta$, GD and preconditioned GD converge linearly: penalty-form steps shrink with the gradient. GD is slow because its stable step is capped by the largest curvature ($\eta<2/25$); preconditioning by $\mathrm{diag}(H)$ equalizes the curvatures (exactly, since $H$ is diagonal) and converges in a few steps. The trust-region methods take steps of fixed length $\eta$, so near the minimum they rattle at distance of order $\eta$ and the loss stalls; sign descent's approach, though, is not slowed by the ill-conditioning. With $\eta_t\propto1/\sqrt{t+1}$ they keep improving, at a rate set by the schedule rather than the curvature, as in the practice of decaying Adam's and Muon's learning rate at the end of training.
"""
    ),
    # ================================================================== 4.2
    markdown(
        r"""
<a id="td24-2"></a>

## 4.2 Natural gradient: invariance, and what it inherits from KL

Gradient descent follows a different path if you write the same model in different
parameters. Natural gradient, $\Delta\theta=-\eta F^{-1}g$ with $F$ the Fisher information,
does not, because $F$ transforms the way a metric must. The same reasoning shows that raw
sharpness $\lambda_{\max}(H)$, with $H$ the loss Hessian, is not a property of the function.

**Paper:** Martens, *New insights and perspectives on the natural gradient method* (see
[Readings](#td24-readings)).

Write $\theta=\phi(\xi)$ for a change of parameters, with invertible Jacobian
$J=\partial\theta/\partial\xi$. Steps are vectors, $\Delta\theta=J\,\Delta\xi$; gradients are
covectors, $g_\xi=J^\top g_\theta$. Gradient descent sets one equal to the other, so in
$\theta$ coordinates its step is $-\eta\,JJ^\top g_\theta$, which depends on the coordinates
you descended in. Turning a covector into a vector needs a metric, $\Delta=-\eta M^{-1}g$
(§4.1's penalty step), and the result is coordinate-free exactly when
$M_\xi=J^\top M_\theta J$. The Fisher, built from score vectors, transforms this way; the
identity matrix of plain GD does not.

<a id="td24-2-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* the model is a distribution $p$ (over outputs, or data); its Fisher information
is $F=\mathbb E_{x\sim p}[\nabla\log p(x)\,\nabla\log p(x)^\top]$, written $F_\theta$ or $F_\xi$
in each coordinate system, and to second order
$\mathrm{KL}(p_\theta\|p_{\theta+\delta})=\tfrac12\delta^\top F\delta$. $g_\theta$, $g_\xi$ are the
loss gradients and $H$ the loss Hessian.

**Where the invariance stops.**
- It is exact for the flow and for affine reparametrizations; for finite steps under a nonlinear $\phi$ it holds only to first order in $\eta$.
- It needs $F\succ0$, but networks have continuous symmetries (scale one ReLU layer up and the next down) that make $F$ singular. Pseudo-inverses or damping $F+\lambda I$ are then extra choices, and damping does not transform as a metric.
- The "empirical Fisher" (squared gradients at the observed labels) transforms correctly but lacks the link to KL and, in general, to curvature. Adam's $v$ is not even that: it averages squared *minibatch-mean* gradients (example gradients $+1$ and $-1$ give second moment 1 but squared mean 0), it divides by $\sqrt v$ rather than $v$, and being diagonal it breaks the invariance.
- In practice $F$ is inverted approximately: K-FAC uses per-layer Kronecker products $A\otimes G$ of the second moments of the layer's inputs and back-propagated gradients. Shampoo (Exercise 4.1.C) also preconditions from both sides, but with $GG^\top$, $G^\top G$ and the power $-1/4$.
- Natural gradient measures steps by KL, so it inherits KL's blindness to output-space geometry (Exercise 4.2.C).

**Sharpness needs a metric.** At a critical point $H_\xi=J^\top H_\theta J$: the Hessian is a
quadratic form, whose eigenvalues mean something only against a metric $M$ (they are those of
$M^{-1}H$). Raw $\lambda_{\max}(H)$ silently uses $M=I$, so rescaling layers can make sharp
minima flat (Dinh et al.; [MDL MDL1 §4, exercise 4.5](../notebooks/mdl1_statistical_learning.html#wb2-flatness)
works the two-layer example). An invariant curvature is the spectrum of $F^{-1}H$. Newton,
whose metric is the Hessian, is invariant only under affine changes, because away from a
critical point the Hessian picks up a term from the curvature of the coordinate change
(Exercises 4.2.B, 4.2.D). The same object governs stability: preconditioned GD with fixed
$P\succ0$ on a quadratic with $H\succ0$ converges iff $\eta\lambda_{\max}(P^{-1}H)<2$, and this
preconditioned sharpness is what adaptive optimizers hold at their edge of stability
([Cohen et al. 2022](https://arxiv.org/abs/2207.14484); [NB2 §3.5](td2_3_edge_of_stability.html#td23-5)).

**Exercise 4.2.A — Invariance of the natural-gradient flow.** Assume $F_\theta\succ0$.

1. Show $F_\xi=J^\top F_\theta J$ and $g_\xi=J^\top g_\theta$.
2. Show that if $\xi(t)$ solves $\dot\xi=-F_\xi^{-1}g_\xi$, then $\theta(t)=\phi(\xi(t))$ solves $\dot\theta=-F_\theta^{-1}g_\theta$. Show the gradient flow $\dot\xi=-g_\xi$ maps to $\dot\theta=-JJ^\top g_\theta$.
3. For a discrete natural-gradient step in $\xi$, show $\phi(\xi+\Delta\xi)=\theta-\eta F_\theta^{-1}g_\theta+O(\eta^2)$, with no $O(\eta^2)$ term when $\phi$ is affine.
"""
    ),
    answer(
        r"""
**1.** The score transforms by the chain rule, $\nabla_\xi\log p=J^\top\nabla_\theta\log p$, so $F_\xi=\mathbb E\big[J^\top\nabla_\theta\log p\,(\nabla_\theta\log p)^\top J\big]=J^\top F_\theta J$. The same chain rule gives $g_\xi=J^\top g_\theta$.

**2.** $\dot\theta=J\dot\xi=-J\,(J^\top F_\theta J)^{-1}J^\top g_\theta=-J\,J^{-1}F_\theta^{-1}J^{-\top}J^\top g_\theta=-F_\theta^{-1}g_\theta$, whatever $\phi$ was. For gradient flow, $\dot\theta=J\dot\xi=-JJ^\top g_\theta$: preconditioned by $JJ^\top$, which depends on the parametrization. Example: $\theta=c\,\xi$ gives $\dot\theta=-c^2g_\theta$, so rescaling a coordinate by $c$ changes its learning rate by $c^2$.

**3.** The step in $\xi$ is $\Delta\xi=-\eta F_\xi^{-1}g_\xi=-\eta J^{-1}F_\theta^{-1}g_\theta$. Taylor expanding, $\phi(\xi+\Delta\xi)=\phi(\xi)+J\Delta\xi+\tfrac12D^2\phi(\xi)[\Delta\xi,\Delta\xi]+\dots=\theta-\eta F_\theta^{-1}g_\theta+O(\eta^2)$. When $\phi$ is affine, $D^2\phi=0$ and the two iterations agree exactly. The $O(\eta^2)$ term can be large when $J$ is nearly singular: a moderate step in $\theta$ can be an enormous step in $\xi$. Code demo 4.2.D shows both effects.
"""
    ),
    markdown(
        r"""
**Exercise 4.2.B — Sharpness depends on the ruler.** With $H_\theta=\partial^2_\theta L$ and
$H_\xi=\partial^2_\xi L$ the Hessians in each coordinate system:

1. Show $\partial^2_\xi L=J^\top H_\theta J+\sum_k(g_\theta)_k\,\nabla^2_\xi\phi_k$. Conclude $H_\xi=J^\top H_\theta J$ at a critical point.
2. At a minimum with $H_\theta\ne0$, show $\lambda_{\max}(H_\xi)$ can be made as large as you like by choosing $J$, but the eigenvalues of $F_\xi^{-1}H_\xi$ equal those of $F_\theta^{-1}H_\theta$.

<details><summary>Hint</summary>

For 2, $J^{-1}(F_\theta^{-1}H_\theta)J$ is similar to $F_\theta^{-1}H_\theta$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\partial_{\xi_a}L=\sum_k(g_\theta)_k\,\partial_a\phi_k$. Differentiating again,
$\partial_{\xi_a}\partial_{\xi_b}L=\sum_{k,l}(H_\theta)_{kl}\,\partial_a\phi_k\,\partial_b\phi_l+\sum_k(g_\theta)_k\,\partial_a\partial_b\phi_k$, which is the claim, since $J_{ka}=\partial_a\phi_k$. At a critical point $g_\theta=0$ and only $J^\top H_\theta J$ survives. Away from one the Hessian does not transform as a tensor: the second term depends on how curved the coordinate change is. The Fisher has no such term; Code demo 4.2.D shows the consequence for Newton.

**2.** With $J=cI$, $H_\xi=c^2H_\theta$, so $\lambda_{\max}$ scales by $c^2$ and can be anything. Layer rescalings in ReLU networks are reparametrizations of this kind: they change $\lambda_{\max}$ without changing the function. Meanwhile $F_\xi^{-1}H_\xi=(J^\top F_\theta J)^{-1}J^\top H_\theta J=J^{-1}F_\theta^{-1}H_\theta J$, which is similar to $F_\theta^{-1}H_\theta$ and has the same eigenvalues. Raw sharpness is curvature measured against a ruler, the Euclidean norm of whatever coordinates you use; reparametrizing changes the ruler and the number, not the function.
"""
    ),
    markdown(
        r"""
**Exercise 4.2.C — What the Fisher inherits from KL.** Blur the point masses of
Exercise 4.0.A, $P_0=\mathcal N(0,s^2)$ and $P_\theta=\mathcal N(\theta,s^2)$, so that the
Fisher information exists, and ask which geometry it encodes. Here
$W_p(P,Q)=\min\big(\mathbb E|X-Y|^p\big)^{1/p}$ over couplings, the Fisher of a 1D family is
$F(\theta)=\mathbb E[(\partial_\theta\log p_\theta)^2]$, and a natural-gradient step is
$\theta\leftarrow\theta-\eta F^{-1}\partial_\theta L$.

1. Show $\mathrm{KL}(P_0\|P_\theta)=\theta^2/(2s^2)$ and $W_1=W_2=|\theta|$.
2. Gradient descent with step $\eta$ on $L_{\rm KL}=\theta^2/(2s^2)$ and on $L_W=\tfrac12W_2^2=\tfrac12\theta^2$: for which $\eta$ does each converge?
3. Check that the Fisher of $\mathcal N(\theta,s^2)$ is $1/s^2$. Take a natural-gradient step on $L_{\rm KL}$ and compare with GD on $L_W$. What happens to the Fisher as $s\to0$, and what does this say about which geometry it encodes?

<details><summary>Hint</summary>

For 1, $\log p_0(x)-\log p_\theta(x)=\big((x-\theta)^2-x^2\big)/(2s^2)$; take the
expectation under $p_0$. A translation is a valid transport map, and no coupling does better
because $\mathbb E|X-Y|\ge|\mathbb E X-\mathbb E Y|$. For 3, the score is
$\partial_\theta\log p_\theta(x)=(x-\theta)/s^2$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathrm{KL}=\mathbb E_{x\sim p_0}\big[(x-\theta)^2-x^2\big]/(2s^2)=\mathbb E[-2\theta x+\theta^2]/(2s^2)=\theta^2/(2s^2)$. For $W_p$: the coupling $Y=X+\theta$ costs exactly $|\theta|$ in $W_1$ and $W_2$, and any coupling has $\mathbb E|X-Y|\ge|\mathbb E X-\mathbb EY|=|\theta|$ (and $(\mathbb E|X-Y|^2)^{1/2}\ge\mathbb E|X-Y|$), so $W_1=W_2=|\theta|$. As $s\to0$ at fixed $\theta$, $\mathrm{KL}\to\infty$, consistent with 4.0.A; $W$ does not depend on $s$.

**2.** On $L_{\rm KL}$, $\theta\leftarrow(1-\eta/s^2)\theta$, which converges iff $0<\eta<2s^2$: the usable step size is tied to the blur. On $L_W$, $\theta\leftarrow(1-\eta)\theta$, which converges iff $0<\eta<2$, whatever $s$ is.

**3.** $F=\mathbb E[(x-\theta)^2/s^4]=1/s^2$. The natural-gradient step is $\theta\leftarrow\theta-\eta\,s^2\cdot\theta/s^2=(1-\eta)\theta$, identical to GD on $L_W$. But the Fisher is the second-order expansion of KL, so it measures how much the probabilities change, not how far mass moves: as $s\to0$ it diverges, and between point masses there is no Fisher metric at all, while the Wasserstein geometry is unchanged. The two agree here only because, for a translated Gaussian, both reduce to rescaling the translation.
"""
    ),
    markdown(
        r"""
**Code demo 4.2.D — GD, natural gradient and Newton under a nonlinear reparametrization.**
Logistic regression on 2D data, $p(y=1\mid x)=\sigma(\theta^\top x)$ with
$\sigma(z)=1/(1+e^{-z})$, trained on the mean negative log-likelihood (NLL). For this model
(canonical link) the Fisher equals the Hessian of the mean NLL, so in $\theta$ coordinates
natural gradient and Newton with the same damped step $\eta$ are the same algorithm. Now
reparametrize $\theta=(\xi_1,e^{\xi_2})$ and run each method in $\xi$: GD with $g_\xi$, natural
gradient with $F_\xi=J^\top F_\theta J$, and Newton with the full $\xi$-Hessian of
Exercise 4.2.B.1. Compare each with its $\theta$-coordinate twin.
"""
    ),
    code(
        r"""
rd = np.random.default_rng(1)                            # own RNG: the data do not depend on earlier cells
Xd = rd.standard_normal((400, 2)) * np.array([1.0, 0.5])
w_true = np.array([1.5, 3.0])
yd = (rd.random(400) < 1 / (1 + np.exp(-Xd @ w_true))).astype(float)

def nll_grad_hess(w):                                    # for logistic regression, Fisher = Hessian
    z = Xd @ w; p = 1 / (1 + np.exp(-z))
    return np.mean(np.logaddexp(0, z) - yd * z), Xd.T @ (p - yd) / len(yd), (Xd * (p * (1 - p))[:, None]).T @ Xd / len(yd)

phi = lambda xi: np.array([xi[0], np.exp(xi[1])])
def run(method, coords, eta, steps=30, th2_0=0.5):
    th = np.array([0.0, th2_0]); xi = np.array([0.0, np.log(th2_0)]); losses = []
    for _ in range(steps):
        w = th if coords == "θ" else phi(xi)
        nll, g, F = nll_grad_hess(w); losses.append(nll)
        if coords == "θ":
            th = th - eta * (g if method == "GD" else np.linalg.solve(F, g))
        else:
            J = np.diag([1.0, np.exp(xi[1])]); gx = J.T @ g
            if method == "GD":
                M = np.eye(2)
            elif method == "natural gradient":
                M = J.T @ F @ J                           # transforms as a metric
            else:                                         # Newton: the xi-Hessian has the extra g * phi'' term
                M = J.T @ F @ J + np.diag([0.0, g[1] * np.exp(xi[1])])
            xi = xi - eta * np.linalg.solve(M, gx)
    return np.array(losses), (th if coords == "θ" else phi(xi))

fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for ax, (method, eta) in zip(axs, [("GD", 2.0), ("natural gradient", 0.5), ("Newton", 0.5)]):
    for coords, ls in [("θ", "-"), ("ξ", "--")]:
        losses, final = run(method, coords, eta)
        ax.plot(losses, ls, label=f"in {coords}: ends at θ = ({final[0]:.2f}, {final[1]:.2f})")
    ax.set_title(f"{method}, η = {eta}"); ax.set_xlabel("step"); ax.legend(fontsize=8)
axs[0].set_ylabel("mean NLL")
plt.tight_layout(); plt.show()
"""
    ),
    answer(
        r"""
*What the plot shows.* GD follows different trajectories in the two coordinate systems ($JJ^\top$ rescales the second coordinate by $e^{2\xi_2}$). Natural gradient in $\xi$ tracks natural gradient in $\theta$ closely, differing only at $O(\eta^2)$ per step, as 4.2.A.3 predicts. Newton in $\xi$ fails: away from the optimum the extra term $g_2e^{\xi_2}$ in $H_\xi$ is negative here and makes $H_\xi$ indefinite, so the Newton step in $\xi_2$ goes uphill and drives $\theta_2=e^{\xi_2}$ toward 0, where the $\xi$-gradient vanishes without $\theta$ being optimal. In $\theta$ coordinates Newton and natural gradient coincide because $F=H$ for this model; in $\xi$ coordinates they come apart, because $F_\xi=J^\top F_\theta J$ holds everywhere while $H_\xi=J^\top H_\theta J$ holds only at critical points.
"""
    ),
    # ================================================================== 4.3
    markdown(
        r"""
<a id="td24-3"></a>

## 4.3 Adam as smoothed sign descent

With its moving averages switched off, Adam is sign descent, the trust-region step in
$\ell_\infty$. With them on, it is sign descent shrunk by a per-coordinate signal-to-noise
factor: a good picture in some regimes and a misleading one in others. Decoupled weight
decay then confines the weights to a box.

**Paper:** Kunstner, Chen, Lavington and Schmidt, *Noise is not the main factor behind the gap
between SGD and Adam on transformers* (see [Readings](#td24-readings)).

**Why Adam beats SGD on transformers.** Not, it seems, because it copes better with
heavy-tailed gradient noise: removing the noise by growing the batch widens the gap, so the
sign-like geometry looks like the main factor. Kunstner et al.'s 2024 follow-up points to
heavy-tailed class imbalance in language data: under gradient descent the loss on rare
tokens falls slowly, while sign-based methods are much less sensitive to class frequency
(shown empirically, and proved for a simplified linear softmax model in continuous time).

<a id="td24-3-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* per coordinate, with gradient $g_t$ at step $t=1,2,\dots$, Adam keeps EMAs
$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ (from
$m_0=v_0=0$), bias-corrects them, $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$,
and steps $\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. **Sign descent** is
$\theta\leftarrow\theta-\eta\,\mathrm{sign}(g)$.

With the EMAs on (Bernstein and Newhouse call their role open), two limits describe Adam,
both for $\beta_1<1$ and many steps:

- **Smoothed sign.** If a coordinate's gradient has mean $\mu$ and variance $\sigma^2$ and $\beta_2$ is close to 1, then $\hat v\approx\mu^2+\sigma^2$ and the *mean* update is about $-\eta\,\mu/\sqrt{\mu^2+\sigma^2}$: the sign of the mean gradient, shrunk by the noise-to-signal ratio (Exercise 4.3.A.4; Balles and Hennig).
- **Slowly varying preconditioner.** With $\beta_2$ close to 1, $\sqrt{\hat v}$ changes slowly, so over a short window Adam is EMA momentum preconditioned by $P=\mathrm{diag}(\sqrt{\hat v})+\epsilon$; Adam's edge of stability is a statement about $\lambda_{\max}(P^{-1}H)$ ([NB2 §3.5](td2_3_edge_of_stability.html#td23-5)).

With $\epsilon=0$, each coordinate's step is bounded by a constant multiple of $\eta$ when
$\beta_1^2<\beta_2$: 1 for $\beta_1=\beta_2$, about 7.3 for the defaults (Exercise 4.3.B).

**Exercise 4.3.A — Sign descent, scale invariance and units.**

1. Show Adam with $\beta_1=\beta_2=0$, $\epsilon=0$ is sign descent.
2. Show that for $\epsilon=0$ and any $\beta_1,\beta_2$, replacing every gradient $g_t$ by $cg_t$ ($c>0$) leaves Adam's updates unchanged. What happens to SGD's?
3. If $\theta$ has units $[\theta]$ and $L$ has units $[L]$, what are the units of $\eta$ in SGD and in Adam? What does this say about choosing Adam's $\eta$?
4. Model a coordinate's gradients as i.i.d. with mean $\mu$ and variance $\sigma^2$, take $\beta_2\to1$ and many steps. Show the mean of $\hat m/\sqrt{\hat v}$ is about $\mu/\sqrt{\mu^2+\sigma^2}$, and its standard deviation about $\sqrt{(1-\beta_1)/(1+\beta_1)}\;\sigma/\sqrt{\mu^2+\sigma^2}$.
"""
    ),
    answer(
        r"""
**1.** With $\beta_1=0$, $m_t=g_t$ and $1-\beta_1^t=1$, so $\hat m_t=g_t$; likewise $\hat v_t=g_t^2$. The update is $-\eta\,g_t/\sqrt{g_t^2}=-\eta\,\mathrm{sign}(g_t)$ per coordinate (taking $0/0=0$).

**2.** $m_t$ is linear in the gradients, so it becomes $c\,m_t$; $v_t$ is quadratic, so it becomes $c^2v_t$, and $\sqrt{c^2v_t}=c\sqrt{v_t}$. The ratio $\hat m_t/\sqrt{\hat v_t}$ is unchanged. SGD's update is multiplied by $c$: a loss multiplied by 10 needs SGD's learning rate divided by 10, and no change for Adam (until $\epsilon$ matters).

**3.** For SGD, $\Delta\theta=-\eta\,\partial L/\partial\theta$, so $[\eta]=[\theta]^2/[L]$. For Adam, $\hat m/\sqrt{\hat v}$ is dimensionless, so $[\eta]=[\theta]$: a distance per step in parameter units, which can be set from the expected scale of the parameters (Kingma and Ba). So Adam's $\eta$ inherits the parametrization: if a weight's natural scale changes with width, so must $\eta$ (§4.5).

**4.** With $\beta_2\to1$, $\hat v$ averages $g^2$ over a long window, so $\hat v\approx\mathbb Eg^2=\mu^2+\sigma^2$, nearly constant, and $\hat m/\sqrt{\hat v}\approx\hat m/\sqrt{\mu^2+\sigma^2}$. For a stationary sequence and many steps, $\mathbb E\hat m=\mu$ and $\mathrm{Var}\,m=(1-\beta_1)^2\sum_k\beta_1^{2k}\sigma^2=\frac{1-\beta_1}{1+\beta_1}\sigma^2$. Dividing by $\sqrt{\mu^2+\sigma^2}$ gives both claims. For $\beta_1=0.9$ the factor $\sqrt{(1-\beta_1)/(1+\beta_1)}\approx0.23$: on a pure-noise coordinate ($\mu=0$) the mean step is zero but individual steps are about $0.23\eta$; on a noiseless one ($\sigma=0$) every step is $\pm\eta$, pure sign descent. Code demo 4.3.C checks both.
"""
    ),
    markdown(
        r"""
**Exercise 4.3.B — A sharp per-coordinate bound (harder).** How large can one coordinate's
step $\eta\,|m_t|/\sqrt{v_t}$ be, over all gradient histories? Ignore bias correction (many
steps) and take $\epsilon=0$, so $m_t=(1-\beta_1)\sum_{k\ge0}\beta_1^kg_{t-k}$ and
$v_t=(1-\beta_2)\sum_{k\ge0}\beta_2^kg_{t-k}^2$.

1. Assuming $\beta_1^2<\beta_2$, show
$$\frac{|m_t|}{\sqrt{v_t}}\le\frac{1-\beta_1}{\sqrt{(1-\beta_2)\,(1-\beta_1^2/\beta_2)}},$$
and find the gradient history that attains it. Evaluate for $\beta_1=0.9$, $\beta_2=0.999$, and for $\beta_1=\beta_2$.
2. Kingma and Ba state that for these defaults the step is bounded by $(1-\beta_1)/\sqrt{1-\beta_2}\approx3.16$ times $\eta$, from a gradient that is zero except at the current step. Compare.

<details><summary>Hint</summary>

Write $\beta_1^k=\big(\beta_1^k\beta_2^{-k/2}\big)\,\beta_2^{k/2}$ and apply Cauchy–Schwarz to
$\sum_k\beta_1^kg_{t-k}$. Equality in Cauchy–Schwarz holds when the two sequences are
proportional.
</details>
"""
    ),
    answer(
        r"""
**1.** By Cauchy–Schwarz,
$$
\Big|\sum_k\beta_1^kg_{t-k}\Big|=\Big|\sum_k\frac{\beta_1^k}{\beta_2^{k/2}}\cdot\beta_2^{k/2}g_{t-k}\Big|
\le\Big(\sum_k\frac{\beta_1^{2k}}{\beta_2^k}\Big)^{1/2}\Big(\sum_k\beta_2^kg_{t-k}^2\Big)^{1/2}
=\frac{1}{\sqrt{1-\beta_1^2/\beta_2}}\sqrt{\frac{v_t}{1-\beta_2}},
$$
where the geometric series converges because $\beta_1^2<\beta_2$. Multiplying by $1-\beta_1$ gives the bound. Equality holds when $\beta_2^{k/2}g_{t-k}\propto\beta_1^k\beta_2^{-k/2}$, i.e. $g_{t-k}\propto(\beta_1/\beta_2)^k$: a gradient that has been growing geometrically, by a factor $\beta_2/\beta_1\approx1.11$ per step. For $\beta_1=0.9$, $\beta_2=0.999$ the bound is $0.1/\sqrt{0.001\times0.18919}\approx7.27$. For $\beta_1=\beta_2=\beta$ it is $(1-\beta)/\sqrt{(1-\beta)(1-\beta)}=1$, attained by a constant gradient. (For $0<\beta_2<1$ there is no history-independent bound once $\beta_1^2\ge\beta_2$: if $\beta_1^2>\beta_2$, $v$ forgets a past spike faster than $m$ does and the ratio grows; at $\beta_1^2=\beta_2$ the weights in the Cauchy–Schwarz sum stop decaying, and ever longer aligned histories make the ratio grow like the square root of their length.)

**2.** A single spike gives $m_t=(1-\beta_1)g$ and $v_t=(1-\beta_2)g^2$, so the ratio is $(1-\beta_1)/\sqrt{1-\beta_2}\approx3.16$. That history is not the worst one, so this is not an upper bound: a steadily growing gradient gives up to $7.27$, because $v$ forgets more slowly than $m$ and lags behind it. Adam's step is an $\ell_\infty$ trust region in a loose sense, with radius up to about $7.3\eta$ for the default $\beta$'s; in practice typical steps are below $\eta$ (4.3.A.4).
"""
    ),
    markdown(
        r"""
**Code demo 4.3.C — Adam's average step against the signal-to-noise prediction.** Feed Adam
i.i.d. gradients $g_t=\mu+\xi_t$, $\xi_t\sim\mathcal N(0,1)$, for a range of $\mu$, and compare
the normalized step $\hat m_t/\sqrt{\hat v_t}$ (update divided by $\eta$, $\epsilon=0$) with
4.3.A.4's mean $\mu/\sqrt{\mu^2+1}$ and standard deviation
$\sqrt{(1-\beta_1)/(1+\beta_1)}/\sqrt{\mu^2+1}$. Then feed it 4.3.B's worst-case history, a
geometrically growing gradient, and compare with that bound.
"""
    ),
    code(
        r"""
def adam_ratio(gs, b1=0.9, b2=0.999, bias_correction=True):
    # vectorized over the first axis: gs has shape (n_runs, T); returns m_hat / sqrt(v_hat) at each step (0 where v = 0)
    m = np.zeros(gs.shape[0]); v = np.zeros(gs.shape[0]); out = np.empty_like(gs)
    for t in range(gs.shape[1]):
        m = b1 * m + (1 - b1) * gs[:, t]; v = b2 * v + (1 - b2) * gs[:, t] ** 2
        cm, cv = (1 - b1 ** (t + 1), 1 - b2 ** (t + 1)) if bias_correction else (1.0, 1.0)
        out[:, t] = np.divide(m / cm, np.sqrt(v / cv), out=np.zeros_like(m), where=v > 0)
    return out

mus = np.linspace(-3, 3, 25)
T = 20000
gs = mus[:, None] + rng.standard_normal((len(mus), T))
r = adam_ratio(gs, b2=0.9999)[:, 2000:]                  # drop the warm-up
mu_f = np.linspace(-3, 3, 300)
sd_pred = np.sqrt(0.1 / 1.9) / np.sqrt(mu_f**2 + 1)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].plot(mu_f, mu_f / np.sqrt(mu_f**2 + 1), "k-", lw=1, label=r"mean: $\mu/\sqrt{\mu^2+\sigma^2}$")
axs[0].fill_between(mu_f, mu_f / np.sqrt(mu_f**2 + 1) - sd_pred, mu_f / np.sqrt(mu_f**2 + 1) + sd_pred,
                    color="0.85", label="± predicted s.d.")
axs[0].plot(mu_f, np.sign(mu_f), color="0.5", ls=":", label="sign descent")
axs[0].errorbar(mus, r.mean(1), yerr=r.std(1), fmt="o", ms=3, lw=0.8, label=r"Adam: $\hat m/\sqrt{\hat v}$, mean ± s.d.")
axs[0].set_xlabel(r"mean gradient $\mu$ (noise $\sigma=1$)"); axs[0].legend(fontsize=7)

b1, b2 = 0.9, 0.999
bound = (1 - b1) / np.sqrt((1 - b2) * (1 - b1**2 / b2))
Tw = 3000
worst = (b1 / b2) ** np.arange(Tw)[::-1]                 # g_{t-k} ∝ (b1/b2)^k
spike = np.zeros(Tw); spike[-1] = 1.0
noise = rng.standard_normal(Tw)
for name, seq in [("growing (worst case)", worst), ("single spike", spike), ("Gaussian noise", noise)]:
    axs[1].plot(np.abs(adam_ratio(seq[None, :], bias_correction=False)[0])[-60:], label=name)
axs[1].axhline(bound, color="k", ls="--", lw=1); axs[1].text(1, bound * 0.92, f"bound {bound:.2f}", fontsize=8)
axs[1].set_xlabel("last 60 of 3000 steps (no bias correction, as in 4.3.B)"); axs[1].set_ylabel(r"$|\hat m/\sqrt{\hat v}|$")
axs[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"max |m/sqrt v| on the growing history: {np.abs(adam_ratio(worst[None, :], bias_correction=False)).max():.3f}"
      f"  (bound {bound:.3f});  s.d. at mu = 0: {r[len(mus) // 2].std():.3f} (predicted {np.sqrt(0.1 / 1.9):.3f})")
"""
    ),
    markdown(
        r"""
**Exercise 4.3.D — Where the sign picture fails.** "Adam is sign descent" is exact only with
its EMAs off. Model a coordinate's gradient as having mean $\mu$ and variance $\sigma^2$. In
each regime below, what is Adam's per-coordinate step $-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$
actually, and what describes it better than sign descent?

- (i) Noisy coordinates: $\sigma\gg|\mu|$.
- (ii) Oscillating gradients: the gradient flips sign every step, as along the sharpest directions at the edge of stability.
- (iii) Tiny gradients: $\sqrt{\hat v}\ll\epsilon$.

<details><summary>Hint</summary>

(i) Use 4.3.A.4. (ii) What do $m$ and $v$ average to, and what is left if you freeze
$\sqrt{\hat v}$? (iii) Expand $\hat m/(\sqrt{\hat v}+\epsilon)$.
</details>
"""
    ),
    answer(
        r"""
- **Noisy coordinates.** The mean signed update is about $\eta|\mu|/\sigma\ll\eta$ (4.3.A.4), with fluctuations of about $0.23\eta$ around it, not steps of $\pm\eta$. "Sign of the mean, weighted by signal-to-noise, plus noise" describes it better.
- **Oscillating gradients.** At the edge of stability the gradient along the top curvature directions flips sign every step ([NB2 §3](td2_3_edge_of_stability.html#td23-1)). Then $m$ averages toward zero while $v$ does not, and the right description is EMA momentum with a slowly varying diagonal preconditioner $P$. With $P\succ0$ frozen, on a quadratic with $H\succ0$ and $0\le\beta_1<1$, the iteration converges for every start iff $\eta\lambda_{\max}(P^{-1}H)<2(1+\beta_1)/(1-\beta_1)$, which is 38 for $\beta_1=0.9$ ([Cohen et al. 2022](https://arxiv.org/abs/2207.14484)).
- **Tiny gradients.** When $\sqrt{\hat v}\ll\epsilon$, for example for rarely used embedding rows, the update is about $-\eta\hat m/\epsilon$ (at $\sqrt{\hat v}=\epsilon$ the denominator is already $2\epsilon$): SGD with momentum and learning rate $\eta/\epsilon$, neither sign-like nor scale-invariant.

A fourth, milder one is a growing gradient, where $v$ lags and steps exceed $\eta$ (4.3.B).
"""
    ),
    markdown(
        r"""
**Exercise 4.3.E — Decoupled weight decay as an implicit constraint.** AdamW's update has the
form $\theta_{t+1}=\theta_t+\eta_t(o_t-\lambda\theta_t)$, with schedule $\eta_t$, weight decay
$\lambda$, and $o_t=-\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ the optimizer's own step direction.
Muon with decoupled weight decay has the same form with $o_t$ an approximately orthogonalized
matrix.

1. Suppose $\|o_t\|\le b$ in some norm and $0<\eta_t\lambda\le1$. Show $\|\theta_{t+1}\|-b/\lambda\le(1-\eta_t\lambda)\big(\|\theta_t\|-b/\lambda\big)$. Conclude that the ball $\|\theta\|\le b/\lambda$ is never left once entered, that it is approached from outside if $\sum_t\eta_t=\infty$, and geometrically fast if $\eta_t$ stays above a positive constant.
2. Apply this to sign descent with decoupled decay ($\ell_\infty$, $b=1$) and to exactly orthogonalized updates (spectral norm, $b=1$). What box or ball do you get?

<details><summary>Hint</summary>

Triangle inequality on $(1-\eta_t\lambda)\theta_t+\eta_to_t$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\|\theta_{t+1}\|=\|(1-\eta_t\lambda)\theta_t+\eta_to_t\|\le(1-\eta_t\lambda)\|\theta_t\|+\eta_tb$. Subtracting $b/\lambda$ from both sides, $\|\theta_{t+1}\|-b/\lambda\le(1-\eta_t\lambda)\|\theta_t\|+\eta_tb-b/\lambda=(1-\eta_t\lambda)(\|\theta_t\|-b/\lambda)$. If $\|\theta_t\|\le b/\lambda$, the right side is $\le0$, so the ball is invariant. From outside, iterating gives an excess of at most $\prod_{s<t}(1-\eta_s\lambda)$ times the initial one. The product tends to 0 if $\sum_s\eta_s=\infty$ (with summable steps and every $\eta_s\lambda<1$ the weights can stop short of the ball), and it decays geometrically if $\eta_s\ge\eta_{\min}>0$. Even then the ball is approached, not necessarily entered in finite time.

**2.** Sign descent has $\|\mathrm{sign}(g)\|_\infty\le1$, so with decoupled decay the weights approach the box $\max_i|\theta_i|\le1/\lambda$. Exactly orthogonalized updates have $\|UV^\top\|_2=1$, so the weights approach the spectral-norm ball $\|W\|_2\le1/\lambda$. Weight decay picks the radius of a constraint set in the same norm the optimizer uses for its steps. For Adam itself the step bound is $b\approx7.3$ rather than 1 (4.3.B), so the box is only loosely $1/\lambda$; practical Muon, with its approximate quintic orthogonalization (singular values up to about $1.2$, §4.4) and shape factor, has a correspondingly adjusted radius.

For idealized versions of these algorithms the constraint picture is a theorem: for Lion ([Chen, Liu, Liang and Liu](https://arxiv.org/abs/2310.05898)), for full-batch AdamW with $\epsilon=0$, whose limits under a non-increasing schedule with $\sum_t\eta_t=\infty$ are KKT points of the loss subject to $\|\theta\|_\infty\le1/\lambda$ ([Xie and Li, ICML 2024](https://arxiv.org/abs/2404.04454); with $\epsilon>0$ AdamW can stall at an interior non-stationary point), and for Muon with exact orthogonalization, via part 1's inequality ([Chen, Li and Liu](https://arxiv.org/abs/2506.15054), Proposition 1).
"""
    ),
    # ================================================================== 4.4
    markdown(
        r"""
<a id="td24-4"></a>

## 4.4 Newton–Schulz: orthogonalizing without an SVD

Muon replaces each hidden weight matrix's momentum by an approximation of $UV^\top$, the
matrix with the same singular vectors and every singular value set to 1, computed by five
steps of an odd matrix polynomial instead of an SVD. Its tuned quintic gives up exact
convergence to grow small singular values fast.

**Paper:** Jordan et al., *Muon: an optimizer for hidden layers* (see [Readings](#td24-readings)).

$UV^\top$ is the steepest-descent direction under the spectral norm (§4.1), which measures
how much an update can change a layer's output (§4.5). Muon orthogonalizes the (Nesterov)
momentum of each 2D hidden-layer weight; embeddings, the output head and vector parameters
get AdamW. The post adds a hypothesis it labels as speculation: SGD-momentum and Adam updates
for transformer weights are nearly low-rank, and orthogonalizing raises the scale of "rare
directions" that matter for learning.

<a id="td24-4-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* for a matrix $M$ with reduced SVD $M=U\Sigma V^\top$, $\mathrm{Ortho}(M)=UV^\top$.
Muon's iteration is $X\leftarrow aX+b(XX^\top)X+c(XX^\top)^2X$, started from $X_0=G/\|G\|_F$
with $G$ the matrix being orthogonalized (the momentum), $(a,b,c)=(3.4445,-4.7750,2.0315)$,
and five steps. On singular values it acts as $\sigma\mapsto\varphi(\sigma)=a\sigma+b\sigma^3+c\sigma^5$.
The classical cubic iteration $X\leftarrow\tfrac32X-\tfrac12XX^\top X$, i.e.
$p(\sigma)=(3\sigma-\sigma^3)/2$, converges to $UV^\top$. Per the post, Muon's coefficients
maximize $a=\varphi'(0)$ subject to the iterates settling in roughly $[0.7,1.3]$ rather than
at 1, after the authors found that errors of about $\pm0.3$ in the singular values do not
hurt training (their motivation, not an established result). The reference implementation
also rescales by $\sqrt{\max(1,\text{fan-out}/\text{fan-in})}$ (Exercise 4.5.A).

**Exercise 4.4.A — Odd polynomials act on singular values.**

1. For $X=U\Sigma V^\top$ (reduced SVD), show $(XX^\top)^kX=U\Sigma^{2k+1}V^\top$. Conclude that $N$ steps produce $U\varphi^{N}(\Sigma)V^\top$, with $\varphi$ applied to each singular value.
2. Why divide by $\|G\|_F$ first, rather than by $\|G\|_2$? What does it cost when $G$ has many comparable singular values?
"""
    ),
    answer(
        r"""
**1.** $XX^\top=U\Sigma V^\top V\Sigma U^\top=U\Sigma^2U^\top$ (using $V^\top V=I$), so for $k\ge1$, $(XX^\top)^k=U\Sigma^{2k}U^\top$ and $(XX^\top)^kX=U\Sigma^{2k}U^\top U\Sigma V^\top=U\Sigma^{2k+1}V^\top$; $k=0$ is trivial. A linear combination of such terms is $U\varphi(\Sigma)V^\top$, with the same singular vectors, and iterating gives $U\varphi^N(\Sigma)V^\top$. (A negative $\varphi^N(\sigma_i)$ would give singular value $|\varphi^N(\sigma_i)|$ with the sign absorbed into $u_i$; for the cubic and the quintic on the ranges used here the values stay positive.) Zero singular values stay zero, which is why the limit is the reduced $UV^\top$ (4.1.C.3).

**2.** The analysis needs every initial singular value in $(0,1]$. $\|G\|_F=\big(\sum\sigma_i^2\big)^{1/2}\ge\sigma_{\max}$ guarantees this at the cost of one cheap reduction, while $\|G\|_2$ would need its own iterative computation. The cost: if $G$ has $r$ comparable singular values, $\|G\|_F\approx\sqrt r\,\sigma_{\max}$, so even the largest singular values start near $1/\sqrt r$ and the small ones much lower. For a $256\times512$ Gaussian matrix, all singular values of $G/\|G\|_F$ lie in about $[0.019,0.105]$. This is why the growth rate near zero, $\varphi'(0)$, is what Muon optimizes.
"""
    ),
    markdown(
        r"""
**Exercise 4.4.B — The cubic iteration.** By 4.4.A the classical iteration acts on each
singular value by a scalar map $p(\sigma)$.

0. Find the odd cubic $p(\sigma)=a\sigma+b\sigma^3$ with $p(1)=1$ and $p'(1)=0$ (1 is a superattracting fixed point).

From now on let $p(\sigma)=(3\sigma-\sigma^3)/2$.

1. Find the fixed points. Show that for $\sigma_0\in(0,1)$ the iterates increase monotonically to 1, and that for $\sigma_0\in(1,\sqrt3)$ the first step lands in $(0,1)$.
2. Write $\sigma=1+e$ and show $e\mapsto-\tfrac32e^2-\tfrac12e^3$. What is the order of convergence?
3. Near 0, $p(\sigma)\approx\tfrac32\sigma$. Roughly how many steps does $\sigma_0=10^{-3}$ need to reach $0.5$?

<details><summary>Hint</summary>

For 1, factor $1-p(\sigma)$ and $p(\sigma)-\sigma$.
</details>
"""
    ),
    answer(
        r"""
**0.** $p(1)=a+b=1$ and $p'(1)=a+3b=0$ give $b=-\tfrac12$, $a=\tfrac32$: $p(\sigma)=\tfrac32\sigma-\tfrac12\sigma^3$, the classical iteration.

**1.** $p(\sigma)=\sigma\iff\sigma(1-\sigma^2)=0$, so the fixed points are $0,\pm1$. For $\sigma\in(0,1)$: $p(\sigma)-\sigma=\tfrac12\sigma(1-\sigma^2)>0$ and $1-p(\sigma)=\tfrac12(1-\sigma)^2(2+\sigma)>0$, so the iterates increase and stay below 1; a bounded increasing sequence converges, and its limit is a fixed point in $(0,1]$, which can only be 1. On $(1,\sqrt3)$, $p'(\sigma)=\tfrac32(1-\sigma^2)<0$, so $p$ decreases from $p(1)=1$ to $p(\sqrt3)=0$ and maps the interval into $(0,1)$. Frobenius normalization keeps every $\sigma_0\le1$, so larger starting values never arise.

**2.** $p(1+e)=\tfrac12\big(3+3e-(1+3e+3e^2+e^3)\big)=1-\tfrac32e^2-\tfrac12e^3$. So $e_{k+1}\approx-\tfrac32e_k^2$: quadratic convergence, from $p'(1)=0$. ("Cubic" refers to the polynomial, not the order.)

**3.** The singular value grows by about $3/2$ per step until it is order one, so it needs about $\log(500)/\log(1.5)\approx15$ steps to reach $0.5$; iterating exactly gives 16. With five steps, only singular values that start above about $0.09$ reach $0.6$. The slow start, not the final convergence, is the bottleneck.
"""
    ),
    markdown(
        r"""
**Exercise 4.4.C — Muon's quintic does not converge to 1.** Muon gives up the cubic's
conditions $p(1)=1$, $p'(1)=0$ in exchange for a large $\varphi'(0)$. What does it get? Parts
2 and 3 need a few lines of numpy.

1. Compute $\varphi(1)$ and $\varphi'(0)$. Is 1 a fixed point?
2. Find the positive fixed points numerically (they solve $c\sigma^4+b\sigma^2+(a-1)=0$) and the derivative of $\varphi$ at the one in $(0,1)$. Is it attracting?
3. Where do starting points in $(0,1]$ end up after 5 steps? (Apply $\varphi$ five times to `np.linspace(1e-4, 1, 10**5)`.)
4. In five steps, how much does $\varphi$ amplify a small singular value, compared with the cubic? What is traded for what?
"""
    ),
    answer(
        r"""
**1.** $\varphi(1)=a+b+c=0.701$ and $\varphi'(0)=a=3.4445$. Since $\varphi(1)\ne1$, the iteration does not converge to $UV^\top$.

**2.** The positive fixed points solve $c\sigma^4+b\sigma^2+(a-1)=0$, a quadratic in $\sigma^2$, giving $\sigma\approx0.868$ and $\sigma\approx1.264$. At the first, $\varphi'(0.868)=a+3b\sigma^2+5c\sigma^4\approx-1.58$: $|\varphi'|>1$, so it repels, and the negative sign means iterates overshoot it from side to side. Numerically, generic starting points never settle: iterated further, they keep cycling through values between about $0.68$ and $1.13$.

**3.** After five steps, starting points in $[10^{-4},1]$ end up in about $[0.048,1.20]$; the top of this range is the maximum of $\varphi$ on $[0,1]$, $\approx1.202$ at $\sigma\approx0.55$. Every starting value above about $1.3\times10^{-3}$ is at least $0.6$ after five steps.

**4.** Near 0 the quintic multiplies by $a=3.4445$ per step, so by $a^5\approx485$ in five steps; the cubic multiplies by $1.5^5\approx7.6$. In five steps the quintic brings singular values from about $10^{-3}$ up to order one, which the cubic needs about 16 steps to do. The price is accuracy: the output is $U\tilde\Sigma V^\top$ with $\tilde\Sigma$ in a band around 1 rather than $UV^\top$ exactly. For an optimizer step, whose overall size is set by $\eta$ anyway, Muon's authors found this an acceptable trade.
"""
    ),
    markdown(
        r"""
**Exercise 4.4.D (code) — Singular values through the iterations.** Apply both iterations,
each started from $G/\|G\|_F$, to a gradient-like matrix $G$ (singular values spread over two
decades, plus a little noise), and plot the histogram of singular values after each step
together with the two scalar maps.
"""
    ),
    code_answer(
        r"""
def newton_schulz(G, coeffs, steps):
    # coeffs = (a, b, c): X <- a X + b (X X^T) X + c (X X^T)^2 X, starting from G / ||G||_F.
    # Return a list of singular-value arrays, one per iterate (including the start).
    raise NotImplementedError

CUBIC = (1.5, -0.5, 0.0)
MUON = (3.4445, -4.7750, 2.0315)
# Build G (e.g. 128 x 256 with singular values spread over two decades), run both
# iterations for 5 steps (and the cubic for 20), and histogram log10 of the singular values.
""",
        r"""
def newton_schulz(G, coeffs, steps):
    a, b, c = coeffs
    X = G / np.linalg.norm(G)
    out = [np.linalg.svd(X, compute_uv=False)]
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
        out.append(np.linalg.svd(X, compute_uv=False))
    return out

CUBIC = (1.5, -0.5, 0.0)
MUON = (3.4445, -4.7750, 2.0315)

m, n = 128, 256
U0, _ = np.linalg.qr(rng.standard_normal((m, m))); V0, _ = np.linalg.qr(rng.standard_normal((n, m)))
G = U0 @ np.diag(np.logspace(0, -2, m)) @ V0.T + 1e-3 * rng.standard_normal((m, n))

cub = newton_schulz(G, CUBIC, 20); quin = newton_schulz(G, MUON, 5)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
s = np.linspace(0, 1.25, 400)
axs[0].plot(s, (3 * s - s**3) / 2, label="cubic p(σ)")
axs[0].plot(s, MUON[0] * s + MUON[1] * s**3 + MUON[2] * s**5, label="Muon quintic φ(σ)")
axs[0].plot(s, s, "k:", lw=1); axs[0].axhspan(0.7, 1.3, color="0.85", alpha=0.5, lw=0)
axs[0].set_xlabel("σ"); axs[0].legend(fontsize=8); axs[0].set_title("scalar maps (band: 0.7–1.3)")
bins = np.linspace(-4, 0.3, 60)
for ax, seq, ks, title in [(axs[1], cub, [0, 5, 10, 20], "cubic"), (axs[2], quin, [0, 1, 2, 3, 5], "Muon quintic")]:
    for k in ks:
        ax.hist(np.log10(seq[k]), bins=bins, histtype="step", lw=1.5, label=f"step {k}")
    ax.set_xlabel("log10 singular value"); ax.set_title(title); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"cubic after 5 steps:   σ in [{cub[5].min():.3f}, {cub[5].max():.3f}]")
print(f"cubic after 20 steps:  σ in [{cub[20].min():.3f}, {cub[20].max():.3f}]")
print(f"quintic after 5 steps: σ in [{quin[5].min():.3f}, {quin[5].max():.3f}]")
""",
    ),
    # ================================================================== 4.5
    markdown(
        r"""
<a id="td24-5"></a>

## 4.5 Width: which step sizes transfer

Under muP (the maximal update parametrization) every layer's features move by order one per
step as width grows, so hyperparameters tuned on a narrow model transfer to a wide one.
Yang, Simon and Bernstein derive muP's hidden-layer rules from one spectral condition, and
each optimizer's norm then fixes its learning-rate exponent.

**Papers:** Yang, Simon and Bernstein, *A spectral condition for feature learning*; Yang et
al., *Tensor Programs V* (Table 3) (see [Readings](#td24-readings)).

Measure activations by their RMS norm, $\|x\|_{\rm RMS}=\|x\|_2/\sqrt n$, so entries of order
one give RMS norm of order one at every width. For layer $l$ with weights
$W_l\in\mathbb R^{n_l\times n_{l-1}}$, the spectral condition (their Condition 1) asks that
every layer and every update stretch order-one activations by an order-one factor:

$$
\|W_l\|_2,\ \|\Delta W_l\|_2=\Theta\Big(\sqrt{n_l/n_{l-1}}\Big),
$$

with $\|\cdot\|_2$ the spectral norm. The bound is tight, not loose by a power of the width,
because gradient updates are low-rank and aligned with the incoming features. Each
optimizer's update norm then gives its learning rate: SGD $\eta_l=\Theta(n_l/n_{l-1})$; Adam
$\eta_l=\Theta(1/\text{fan-in})$, since $\mathrm{sign}(G)$ of a low-rank $G$ has spectral norm
as large as its Frobenius norm; Muon, whose $UV^\top$ has spectral norm 1, a factor
$\sqrt{n_l/n_{l-1}}$.

<a id="td24-5-walkthrough"></a>

**Derivations and walkthrough.** *Notation:* layer $l$ has width $n_l$ ($n_0$ is the input dimension), weights $W_l$ (fan-in
$n_{l-1}$, fan-out $n_l$) and features $h_l=W_lh_{l-1}$; $\|\cdot\|_F$ is the Frobenius norm;
$\Theta(\cdot)$ and $O(\cdot)$ refer to width $\to\infty$ at fixed aspect ratios. A matrix
inherits the operator norm
$\|W\|_{\rm RMS\to RMS}=\max_x\|Wx\|_{\rm RMS}/\|x\|_{\rm RMS}=\sqrt{n_{\rm in}/n_{\rm out}}\,\|W\|_2$
(Exercise 4.5.A), so Condition 1 reads $\|W_l\|_{\rm RMS\to RMS},\|\Delta W_l\|_{\rm RMS\to RMS}=\Theta(1)$.

Yang–Simon–Bernstein define feature learning as $\|h_l\|_{\rm RMS}=\Theta(1)$ and
$\|\Delta h_l\|_{\rm RMS}=\Theta(1)$ at every width; Condition 1 is sufficient for it, and any
larger scaling blows up as width grows. The bounds come from $\|Wh\|\le\|W\|\,\|h\|$ and are
attained because a batch-size-1 update is an outer product $\delta\,h_{l-1}^\top$, with
$\delta=\partial L/\partial h_l$ the back-propagated error, whose top right singular vector
*is* $h_{l-1}$ (Exercise 4.5.B). The learning-rate rules follow: SGD in Exercise 4.5.B.4, Adam
in Exercise 4.5.C (where a *dense random* sign matrix would instead give $1/\sqrt n$: the
gradient's structure sets the exponent), Muon in Exercise 4.5.A.2. All of this assumes updates
have low **stable rank** $\|A\|_F^2/\|A\|_2^2$ (at most the rank, ignoring tiny singular
values) and stay aligned with the incoming features even when many examples contribute.

<details><summary>When does the alignment assumption hold?</summary>

For one batch of $B\ll n$ examples, $G=\sum_{i\le B}\delta_ih_i^\top$ has rank at most $B$.
But Adam's EMA accumulates many batches, and transformer batches hold far more tokens than
the width, so "rank $\le B$" says nothing there. Yang–Simon–Bernstein measure low stable rank
(below 10 for width 300) and strong alignment of MLP updates on CIFAR-10 even at large batch.
For Adam they rely on the Tensor Programs picture (their Appendix B). That muP's learning
rates transfer in large transformers is an empirical finding of Tensor Programs V,
consistent with this calculation but not implied by it.
</details>

**Exercise 4.5.A — Layers in RMS units.** Let $W=U\Sigma V^\top$ be a reduced SVD.

1. Show $\|W\|_{\rm RMS\to RMS}=\sqrt{n_{\rm in}/n_{\rm out}}\,\|W\|_2$ for $W\in\mathbb R^{n_{\rm out}\times n_{\rm in}}$, and check that Condition 1's two forms agree.
2. Show the trust-region steepest-descent step of radius $\eta$ in $\|\cdot\|_{\rm RMS\to RMS}$ is $-\eta\sqrt{n_{\rm out}/n_{\rm in}}\,UV^\top$. Compare with Muon's reference implementation, which multiplies its orthogonalized update by $\sqrt{\max(1,n_{\rm out}/n_{\rm in})}$.
"""
    ),
    answer(
        r"""
**1.** $\|Wx\|_{\rm RMS}/\|x\|_{\rm RMS}=\big(\|Wx\|_2/\sqrt{n_{\rm out}}\big)\big/\big(\|x\|_2/\sqrt{n_{\rm in}}\big)=\sqrt{n_{\rm in}/n_{\rm out}}\;\|Wx\|_2/\|x\|_2$; maximizing over $x$ gives $\sqrt{n_{\rm in}/n_{\rm out}}\,\|W\|_2$. So $\|W_l\|_{\rm RMS\to RMS}=\Theta(1)$ is the same as $\|W_l\|_2=\Theta(\sqrt{n_l/n_{l-1}})$, with $n_{\rm in}=n_{l-1}$, $n_{\rm out}=n_l$.

**2.** The norm is a constant multiple of the spectral norm, so it has the same steepest direction, $UV^\top$ (4.1.C), and the unit ball $\|D\|_{\rm RMS\to RMS}\le1$ is $\|D\|_2\le\sqrt{n_{\rm out}/n_{\rm in}}$. The trust-region step is therefore $-\eta\sqrt{n_{\rm out}/n_{\rm in}}\,UV^\top$. Muon's factor $\sqrt{\max(1,n_{\rm out}/n_{\rm in})}$ agrees when the layer widens or is square, and differs by the constant $\sqrt{n_{\rm in}/n_{\rm out}}$ when it narrows; at a fixed aspect ratio that does not change the width scaling.

Moonlight uses a different convention: it scales the full-rank $UV^\top$ of an $A\times B$ matrix (RMS entry $1/\sqrt{\max(A,B)}$) by $0.2\sqrt{\max(A,B)}$, so every update has RMS entry about $0.2$, like AdamW's, and AdamW's hyperparameters can be reused. Its spectral norm $0.2\sqrt{\max(A,B)}$ grows with width for a square layer: a choice about matching AdamW at a given size, not the width scaling of Condition 1.
"""
    ),
    markdown(
        r"""
**Exercise 4.5.B — Where the spectral condition comes from.** "Entries of order one" means
$\|h_l\|_2=\Theta(\sqrt{n_l})$. For one example, the gradient with respect to $W_l$ is
$\delta\,h_{l-1}^\top$, with $\delta=\partial L/\partial h_l$.

1. Assume $\|h_{l-1}\|_2=\Theta(\sqrt{n_{l-1}})$. Show that $\|W_l\|_2=O(\sqrt{n_l/n_{l-1}})$ and $\|\Delta W_l\|_2=O(\sqrt{n_l/n_{l-1}})$ suffice to keep $\|h_l\|$ and the direct contribution $\Delta W_l\,h_{l-1}$ to $\Delta h_l$ at most $O(\sqrt{n_l})$, and that they are needed if this must hold for every input direction. How should a Gaussian initialization $W_l\sim\mathcal N(0,\sigma_l^2)$ scale?
2. For a batch-size-1 update $\Delta W=-\eta\,\delta\,h^\top$, show $\|\Delta W\,h\|_2=\|\Delta W\|_2\,\|h\|_2$. Why does this make the scaling of the direct contribution tight?
3. Contrast: suppose $\Delta W=s\,uv^\top$ is rank one with unit $u,v$, but $v$ is a random direction independent of $h$. How large is $\|\Delta W\,h\|$ typically?
4. *SGD's rule.* Assume each layer's update decreases the loss, to first order, by an amount of order one, and that gradient and update are both low-rank and aligned, so $|\langle\Delta W_l,\nabla_{W_l}L\rangle|=\Theta(\|\Delta W_l\|_2\,\|\nabla_{W_l}L\|_2)$. Derive $\|\nabla_{W_l}L\|_2$ and the SGD learning rate that satisfies Condition 1.

<details><summary>Hint</summary>

For the initialization in 1, a Gaussian $n_l\times n_{l-1}$ matrix with entry variance
$\sigma^2$ has $\|W\|_2\approx\sigma(\sqrt{n_l}+\sqrt{n_{l-1}})$. For 4, write
$\Delta L\approx\langle\nabla_{W_l}L,\Delta W_l\rangle$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\|h_l\|=\|W_lh_{l-1}\|\le\|W_l\|_2\|h_{l-1}\|=\|W_l\|_2\,\Theta(\sqrt{n_{l-1}})$, which is $O(\sqrt{n_l})$ if $\|W_l\|_2=O(\sqrt{n_l/n_{l-1}})$; likewise for $\Delta W_l\,h_{l-1}$. Conversely, if the bound must hold for every input, including one aligned with the top right singular vector of $W_l$ (where the inequality is an equality), the conditions are necessary. For the initialization, $\sigma_l(\sqrt{n_l}+\sqrt{n_{l-1}})=\Theta(\sqrt{n_l/n_{l-1}})$ gives $\sigma_l=\Theta\big(\tfrac1{\sqrt{n_{l-1}}}\min(1,\sqrt{n_l/n_{l-1}})\big)$, Yang–Simon–Bernstein's Parametrization 1; the $\min$ matters only when the layer narrows. The full change $\Delta h_l=\Delta W_lh_{l-1}+W_l\Delta h_{l-1}+\Delta W_l\Delta h_{l-1}$ also carries the previous layer's change forward; the other two terms are controlled by the same bounds recursively, and the paper assumes the terms do not cancel.

**2.** $\Delta W\,h=-\eta\,\delta\,(h^\top h)=-\eta\|h\|^2\delta$, so $\|\Delta W\,h\|=\eta\|\delta\|\|h\|^2$. Also $\|\Delta W\|_2=\eta\|\delta\|\|h\|$ for a rank-one matrix. Hence $\|\Delta W\,h\|=\|\Delta W\|_2\|h\|$: the inequality is an equality, because $h$ is exactly the top right singular vector of the update it produced. So the direct contribution is as large as $\|\Delta W\|_2$ allows, and its scaling is set by $\|\Delta W\|_2$.

**3.** $\|\Delta W\,h\|=s\,|v\cdot h|$. For a random unit $v\in\mathbb R^{n_{l-1}}$ independent of $h$, $v\cdot h$ has mean zero and variance $\|h\|^2/n_{l-1}$, so $\|\Delta W\,h\|\approx\|\Delta W\|_2\,\|h\|/\sqrt{n_{l-1}}$: smaller than the bound by a factor $\sqrt{n_{l-1}}$. A low-rank update affects the features at full strength only because it is aligned with them, and gradient updates are aligned by construction. This is why the spectral norm, and not, say, the Frobenius norm, measures a gradient update's effect.

**4.** Layer $l$'s first-order contribution is $\langle\nabla_{W_l}L,\Delta W_l\rangle=-\eta_l\|\nabla_{W_l}L\|_F^2<0$, and by assumption its magnitude is $\Theta(1)=\Theta(\|\Delta W_l\|_2\|\nabla_{W_l}L\|_2)$. (The assumption is per layer: a total change of order one would not by itself force every layer's share to be of order one.) With $\|\Delta W_l\|_2=\Theta(\sqrt{n_l/n_{l-1}})$ this forces $\|\nabla_{W_l}L\|_2=\Theta(\sqrt{n_{l-1}/n_l})$. An SGD step has $\|\Delta W_l\|_2=\eta_l\|\nabla_{W_l}L\|_2$, so Condition 1 needs $\eta_l=\Theta\big(\sqrt{n_l/n_{l-1}}\big/\sqrt{n_{l-1}/n_l}\big)=\Theta(n_l/n_{l-1})$. For a square hidden layer this is $\Theta(1)$; for the input layer (fixed $n_0$) it grows like the width; for the output layer (fixed $n_L$) it shrinks like $1/\text{width}$. These are muP's SGD rates (TP-V, Table 3) and Yang–Simon–Bernstein's eqs. (18)–(20).
"""
    ),
    markdown(
        r"""
**Exercise 4.5.C — The spectral norm of $\mathrm{sign}(G)$.** A sign-based update is
$\Delta W=-\eta\,\mathrm{sign}(G)$, entrywise. To size it under Condition 1 ($\Theta(1)$ for a
square hidden layer) you need $\|\mathrm{sign}(G)\|_2$, which depends on the structure of $G$.
Assume $G=\sum_{k=1}^ra_kb_k^\top\in\mathbb R^{n\times n}$ with the rank $r$ fixed as
$n\to\infty$ and all entries of $a_k,b_k\in\mathbb R^n$ i.i.d. $\mathcal N(0,1)$ (Gaussian for
definiteness; what matters is factors whose entries are i.i.d. and spread out).

1. For $r=1$, show $\mathrm{sign}(G)=\mathrm{sign}(a)\,\mathrm{sign}(b)^\top$ and $\|\mathrm{sign}(G)\|_2=n$ exactly.
2. *(Harder.)* For general fixed $r$, show $\|\mathrm{sign}(G)\|_2\le n$, and $\|\mathrm{sign}(G)\|_2\ge c_r\,n\,(1+o(1))$ with $c_r=\mathbb E\big[X\,\mathrm{sign}(X+Y)\big]>0$, where $X=a_{11}b_{11}$ and $Y=\sum_{k\ge2}a_{1k}b_{1k}$. Conclude $\|\mathrm{sign}(G)\|_2=\Theta(n)$.
3. Contrast with a dense matrix of i.i.d. random signs, $\|S\|_2\approx2\sqrt n$ (stated; the edge of the Marchenko–Pastur law). What learning-rate scalings for a sign-based optimizer on an $n\times n$ hidden layer follow from each case and Condition 1? Redo it for an $n_l\times n_{l-1}$ layer.

<details><summary>Hint</summary>

For the lower bound in 2, test $\mathrm{sign}(G)$ against $u=a_1/\|a_1\|$ and
$v=b_1/\|b_1\|$ and use a law of large numbers on $\sum_{ij}a_{i1}b_{j1}\,\mathrm{sign}(G_{ij})$.
For $c_r>0$, condition on $X$ and use that $Y$ is symmetric.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathrm{sign}(a_ib_j)=\mathrm{sign}(a_i)\,\mathrm{sign}(b_j)$, so $\mathrm{sign}(G)=s\,t^\top$ with $s,t\in\{\pm1\}^n$ (almost surely no zeros). A rank-one matrix has spectral norm $\|s\|\|t\|=\sqrt n\sqrt n=n$. For an $m\times n$ matrix it is $\sqrt{mn}$, equal to its Frobenius norm.

**2.** *Upper bound:* $\|S\|_2\le\|S\|_F=\sqrt{n^2}=n$ for any $\pm1$ matrix. *Lower bound:* $\|S\|_2\ge u^\top Sv$ for unit $u,v$. With $u=a_1/\|a_1\|$, $v=b_1/\|b_1\|$,
$$
u^\top Sv=\frac{\sum_{ij}a_{i1}b_{j1}\,\mathrm{sign}(G_{ij})}{\|a_1\|\,\|b_1\|}.
$$
The numerator is a sum of $n^2$ terms, each distributed as $X\,\mathrm{sign}(X+Y)$, with mean $c_r$. The terms are dependent (entries in the same row share $a_i$), but only weakly, and a law of large numbers gives $c_rn^2(1+o(1))$; we do not make this rigorous here. The denominator is $n(1+o(1))$. So $\|S\|_2\ge c_rn(1+o(1))$. Positivity: conditional on $X=x>0$, $\mathbb E[\mathrm{sign}(x+Y)]=P(Y>-x)-P(Y<-x)=P(|Y|<x)$ by the symmetry of $Y$, and similarly for $x<0$, so $\mathbb E[X\,\mathrm{sign}(X+Y)\mid X]=|X|\,P(|Y|<|X|)\ge0$, strictly positive with positive probability. For $r=1$, $Y=0$ and $c_1=\mathbb E|a||b|=2/\pi$. Numerically $c_4\approx0.37$ and $c_{16}\approx0.20$. Together, $c_rn\lesssim\|S\|_2\le n$.

**3.** For an $n\times n$ hidden layer Condition 1 asks for $\|\Delta W\|_2=\Theta(1)$. A sign-based update has $\|\Delta W\|_2=\eta\,\Theta(n)$ when $G$ is low-rank, so $\eta=\Theta(1/n)$; it would have $\eta\cdot2\sqrt n$ if the sign matrix were dense and random, giving $\eta=\Theta(1/\sqrt n)$. For an $n_l\times n_{l-1}$ layer the low-rank case gives $\eta\sqrt{n_ln_{l-1}}=\Theta(\sqrt{n_l/n_{l-1}})$, i.e. $\eta=\Theta(1/n_{l-1})=\Theta(1/\text{fan-in})$: muP's Adam rule. The dense case has $\|S\|_2\approx\sqrt{n_l}+\sqrt{n_{l-1}}$, giving $\eta=\Theta\big(\sqrt{n_l/n_{l-1}}/(\sqrt{n_l}+\sqrt{n_{l-1}})\big)$, which is $\Theta(n^{-1/2})$ at a fixed aspect ratio. A width-independent Adam learning rate (standard parametrization) makes hidden-layer updates grow like the width.
"""
    ),
    markdown(
        r"""
**Exercise 4.5.D (code) — $\lVert\mathrm{sign}(G)\rVert_2$ against width.** With a power
iteration, compute $\|\mathrm{sign}(G)\|_2$ for $n\times n$ matrices $G$ of rank $1$, $4$, $16$
(products of Gaussian $n\times r$ and $r\times n$ factors) and dense Gaussian $G$, for $n$ from
64 to 2048, and fit log–log slopes. Compare with $2\sqrt n$ (dense) and with the large-$n$
limit (stated, via the Funk–Hecke formula)
$\|\mathrm{sign}(G)\|_2/n\to\mathbb E|w_1|=\Gamma(r/2)/\big(\sqrt\pi\,\Gamma(\tfrac{r+1}2)\big)$,
$w$ uniform on the unit sphere in $\mathbb R^r$: $1$, $0.424$, $0.203$ for $r=1,4,16$. The
rank-16 slope is still visibly below 1 at these widths, because the finite-$n$ excess over the
limit is still decaying.
"""
    ),
    code_answer(
        r"""
def top_singular_value(S, iters=40):
    # power iteration on S^T S
    raise NotImplementedError

ns = [64, 128, 256, 512, 1024, 2048]
# for r in (1, 4, 16, "dense"): build G, compute top_singular_value(np.sign(G)), store, plot on log-log axes
""",
        r"""
from scipy.special import gammaln

def top_singular_value(S, iters=40):
    v = rng.standard_normal(S.shape[1]).astype(S.dtype)
    for _ in range(iters):
        u = S @ v; u /= np.linalg.norm(u)
        v = S.T @ u; s = np.linalg.norm(v); v /= s
    return s

ns = np.array([64, 128, 256, 512, 1024, 2048])
limit = lambda r: np.exp(gammaln(r / 2) - gammaln((r + 1) / 2)) / np.sqrt(np.pi)   # E|w_1|
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for r in (1, 4, 16, "dense"):
    norms = []
    for n in ns:
        G = rng.standard_normal((n, n)) if r == "dense" else rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
        norms.append(top_singular_value(np.sign(G).astype(np.float32)))
    norms = np.array(norms)
    slope = np.polyfit(np.log(ns), np.log(norms), 1)[0]
    line, = ax.loglog(ns, norms, "o-", ms=4, label=f"rank {r}: slope {slope:.2f}" if r != "dense" else f"dense: slope {slope:.2f}")
    ref = limit(r) * ns if r != "dense" else 2 * np.sqrt(ns)
    ax.loglog(ns, ref, ":", color=line.get_color(), lw=1)
    print(f"{str(r):>5}: ||sign(G)||_2 / limit at n = 2048: {norms[-1] / ref[-1]:.3f}")
ax.set_xlabel("width n"); ax.set_ylabel(r"$\|\mathrm{sign}(G)\|_2$")
ax.set_title("dotted: $\\mathbb{E}|w_1|\\,n$ (low rank), $2\\sqrt{n}$ (dense)", fontsize=9)
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""",
    ),
    markdown(
        r"""
**Code demo 4.5.E — Does the hidden-layer learning rate need to scale?** Ten Adam steps on a
three-layer tanh MLP at widths 64 to 2048, with the middle $n\times n$ layer $W_2$ initialized
at spectral norm $\Theta(1)$ and given either a fixed learning rate or one $\propto1/n$ (the
other layers use muP's Adam rates). The plot shows the total change $\|\Delta W_2\|_2$ and the
RMS of its direct effect $\Delta W_2\,h_1$ on the pre-activations ($h_1$ the initial
first-layer features, 64 inputs). Prediction (Exercise 4.5.C): with a fixed rate both grow
roughly like $n$; with $1/n$ they stay flat. This is about update size, not final
performance; whether the optimal rate transfers is Tensor Programs V's empirical question.
"""
    ),
    code(
        r"""
import torch
torch.set_num_threads(4)

def hidden_update_size(n, lr_hidden, steps=10, B=16, d=16, seed=0, lr_other=1e-2):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(256, d, generator=gen)
    y = torch.sin(2 * X @ torch.randn(d, generator=gen) / d**0.5).unsqueeze(1)
    W1 = (torch.randn(n, d, generator=gen) / d**0.5).requires_grad_()
    W2 = (torch.randn(n, n, generator=gen) / n**0.5).requires_grad_()     # ||W2||_2 ≈ 2 at every width
    W3 = (torch.randn(1, n, generator=gen) / n).requires_grad_()          # muP-style small output layer
    opt = torch.optim.Adam([{"params": [W1], "lr": lr_other},             # input layer: Adam lr Θ(1) in muP
                            {"params": [W2], "lr": lr_hidden},
                            {"params": [W3], "lr": lr_other * 64 / n}])   # output layer: Adam lr ∝ 1/fan-in
    with torch.no_grad():
        h1 = torch.tanh(X[:64] @ W1.T)                                    # initial first-layer features
    W2_0 = W2.detach().clone()
    for _ in range(steps):
        idx = torch.randint(0, 256, (B,), generator=gen)
        out = torch.tanh(torch.tanh(X[idx] @ W1.T) @ W2.T) @ W3.T
        loss = ((out - y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dW = W2 - W2_0
        return torch.linalg.matrix_norm(dW, ord=2).item(), (h1 @ dW.T).pow(2).mean().sqrt().item()

widths = [64, 128, 256, 512, 1024, 2048]
res = {"fixed lr = 1e-3": [hidden_update_size(n, 1e-3) for n in widths],
       "lr = 1e-3 · 64/n": [hidden_update_size(n, 1e-3 * 64 / n) for n in widths]}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for label, vals in res.items():
    for ax, k in zip(axs, (0, 1)):
        y_ = np.array([v[k] for v in vals])
        slope = np.polyfit(np.log(widths), np.log(y_), 1)[0]
        ax.loglog(widths, y_, "o-", ms=4, label=f"{label} (slope {slope:.2f})")
axs[0].set_ylabel(r"$\|\Delta W_2\|_2$ after 10 steps"); axs[1].set_ylabel(r"RMS of $\Delta W_2\,h_1$")
for ax in axs:
    ax.set_xlabel("width n"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    ),
    # ================================================================== 4.6
    markdown(
        r"""
<a id="td24-6"></a>

## 4.6 The modular norm: combining layers, and depth

§4.5 sized each layer's step on its own. The modular norm is one norm for the whole network,
built recursively from its modules, and steepest descent in it moves every layer by its own
steepest step, with radii set by the architecture: $\eta/L$ per layer for a chain of $L$
equal layers.

**Paper:** Large et al., *Scalable optimization in the modular norm* (see
[Readings](#td24-readings)).

Each module (a linear layer, an embedding, a nonlinearity, or a composite) carries a norm on
its weights, a **mass** (its share of the network's learning) and a **sensitivity** (how much
it amplifies input perturbations).

<a id="td24-6-walkthrough"></a>

**Derivations and walkthrough.** Large et al.'s Linear module computes $\sqrt{d_{\rm out}/d_{\rm in}}\,Wx$, with $d_{\rm in}$,
$d_{\rm out}$ its input and output widths, so its plain spectral norm plays the role of §4.5's
RMS→RMS norm; it has mass 1 and sensitivity 1. Composing $M=M_2\circ M_1$ adds masses,
multiplies sensitivities and sets

$$
\|(w_1,w_2)\|_M=\max\Big(\,s_2\,\frac{\mu_1+\mu_2}{\mu_1}\,\|w_1\|_{M_1},\ \frac{\mu_1+\mu_2}{\mu_2}\,\|w_2\|_{M_2}\Big),
$$

with $\mu_i>0$ the masses and $s_2$ the sensitivity of the second module (their
Definition 3, which drops a term when its module has zero mass, as parameter-free
nonlinearities do; concatenation is the same without $s_2$). The factor $s_2$ is there because
a change in the first module's weights reaches the output only through the second module;
the mass ratios split one unit of step budget between the modules. The result is a max of
scaled layer norms, as in §4.1's remark 3. The paper shows the loss is Lipschitz-smooth in
this norm with computable constants when modules are well-normed (for a Linear module,
$\|W\|_2\le1$ and inputs with $\|x\|_{\rm RMS}\le1$), a regime it initializes in but does not
enforce during training.

**Exercise 4.6.A — A chain of layers.** Take a chain of $L$ Linear modules, each with norm
$\|W_l\|_2$, mass 1 and sensitivity 1. Show the modular norm is $L\max_l\|W_l\|_2$. What does
a unit-radius steepest-descent step do to each layer, and what does this worst-case bound
suggest for how step sizes should scale with depth?
"""
    ),
    answer(
        r"""
Let $N_k$ be the norm of the first $k$ layers composed, with mass $k$ and sensitivity 1. Then $N_1=\|W_1\|_2$ and, by the rule with $\mu_1=k-1$, $\mu_2=1$, $s_2=1$,
$$N_k=\max\Big(\tfrac{k}{k-1}N_{k-1},\ k\,\|W_k\|_2\Big).$$
If $N_{k-1}=(k-1)\max_{l<k}\|W_l\|_2$, this gives $N_k=k\max_{l\le k}\|W_l\|_2$, so by induction $N_L=L\max_l\|W_l\|_2$. A max of layer norms is the case of §4.1's remark 3, so the steepest-descent step of modular radius $\eta$ moves every layer by its own steepest step with spectral size $\eta/L$.

The bound behind it is a worst case: inside the well-normed region, a perturbation of size $\epsilon$ in each of $L$ layers changes the output by up to about $L\epsilon$ if the perturbations add coherently. Steps of $\eta/L$ per layer keep that of order $\eta$, which suggests per-layer step sizes (or branch multipliers) shrinking like $1/L$. For residual networks, posed as the scaling $L^{-\alpha}$ of each branch, this is debated: [Tensor Programs VI](https://arxiv.org/abs/2310.02244) and [Bordelon et al.](https://arxiv.org/abs/2309.16620) use $\alpha=\tfrac12$, while [CompleteP (Dey et al. 2025)](https://arxiv.org/abs/2505.01618) reports better hyperparameter transfer with $\alpha=1$ in pre-LN transformers. Masses let some layers (embeddings, say) take a larger or smaller share than the equal split.
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, STEM, STEM))
