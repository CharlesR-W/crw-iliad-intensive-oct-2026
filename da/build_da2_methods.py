#!/usr/bin/env python3
"""Build DA2, "Methods: an ordered tour" (Data Attribution day).

Shape (DA_SPECS_v1.md, DA2 section): an opening with the conventions and the tour order; six
`##` stops with anchors da2-stop-1 ... da2-stop-6, in spec order. Each stop opens with a short
capsule that everyone reads, then its readings (`reading()`, with "Claude-suggested extras"
last), then optional folded `### Walkthrough: ...` subsections (our annotated derivations, an
alternative to reading the paper's derivation) and `### Code: ...` subsections.

Every closed-form result is checked in checks/da2_methods_checks.py; the review log is
../reviews/da/da2_methods.md. Source: Louis Jaburi's Iliad notes (source/data-attribution.tex,
CC BY 4.0), with the errata of DA_SPECS_v1.md applied.
"""
from nbkit_da import answer, code, markdown, reading, run

DA1 = "da1_core.html"

# ======================================================================= readings
GROSSE = reading(
    "Studying Large Language Model Generalization with Influence Functions",
    "https://arxiv.org/abs/2308.03296",
    "Grosse et al. (Anthropic, 2023)",
    "IFs for language models up to 52B parameters; 119 pages, so read selectively.\n\n"
    "*Method* (the walkthrough below covers it): §2.1.1, the PBRF framing; §2.2.2–2.2.3, K-FAC and "
    "EK-FAC; §3.1–3.2, fitting EK-FAC to a transformer's MLP layers, TF-IDF filtering, query batching.\n\n"
    "*Findings* (no derivation replaces these): §5.2.1, influence is heavy-tailed yet spread over many "
    "sequences; §5.3.1 with Figs. 1 and 15–16, influential sequences grow more abstract with scale, "
    "e.g. across languages; §5.3.5, role-play, e.g. self-preservation texts behind the shutdown query.",
    importance=5, length=4)
KFAC = reading(
    "Optimizing Neural Networks with Kronecker-factored Approximate Curvature",
    "https://arxiv.org/abs/1503.05671",
    "Martens and Grosse (ICML 2015)",
    "The origin of K-FAC. §3 and §3.1 state the factorization and the independence assumption behind it; "
    "§5 samples the targets from the model, which is what makes it the Gauss–Newton/Fisher matrix.",
    importance=2, length=3)
EKFAC = reading(
    "Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis",
    "https://arxiv.org/abs/1806.03884",
    "George, Laurent, Bouthillier, Ballas and Vincent (NeurIPS 2018)",
    "EK-FAC: keep K-FAC's eigenbasis, refit the eigenvalues (§3.2), with the Frobenius-optimality proof "
    "in Appendix A.1.",
    importance=2, length=2)
LOGRA = reading(
    "What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions",
    "https://arxiv.org/abs/2405.13954",
    "Choe et al. (2024)",
    "LoGra: project each layer's gradient with a Kronecker-structured random projection built from the "
    "forward and backward activations, so per-example gradients are never formed; runs on 8B models.",
    importance=2, length=3)

SOURCE = reading(
    "Training Data Attribution via Approximate Unrolled Differentiation",
    "https://arxiv.org/abs/2405.12186",
    "Bae, Lin, Lorraine and Grosse (2024)",
    "SOURCE. Read §3.2 (segments, stationarity, the filter $F_r$ and its comparison with damping, Fig. 3) "
    "and §3.3 (chaining segments). Walkthrough 2B derives the same filter. SOURCE scores the effect of "
    "*removing* an example, so its sign is opposite to our upweighting influence.",
    importance=4, length=3)
MAGIC = reading(
    "MAGIC: Near-Optimal Data Attribution for Deep Learning",
    "https://arxiv.org/abs/2504.16430",
    "Ilyas and Engstrom (2025)",
    "Exact derivatives of the training run by reverse mode with replay. Read §4 and Fig. 4: LDS near 1 "
    "(0.96 on CIFAR-10) when about 1% of the data is dropped, with the seed and data order fixed "
    "(§2.1), degrading as the drop fraction grows toward 20%.",
    importance=4, length=2)
TRACIN = reading(
    "Estimating Training Data Influence by Tracing Gradient Descent",
    "https://arxiv.org/abs/2002.08484",
    "Pruthi, Liu, Sundararajan and Kale (NeurIPS 2020)",
    "TracIn, §3: the ideal score is the total decrease of the test loss over the steps that used the "
    "example; its first-order form is $\\sum_t\\eta_t\\nabla\\ell(w_t,z')\\cdot\\nabla\\ell(w_t,z)$ "
    "(divided by $b$ for minibatches), and TracInCP replaces the steps by a few checkpoints.",
    importance=3, length=2)
ENGSTROM = reading(
    "Optimizing ML Training with Metagradient Descent",
    "https://arxiv.org/abs/2503.13751",
    "Engstrom et al. (2025)",
    "The REPLAY algorithm behind MAGIC: exact gradients through training with logarithmic memory, and "
    "when training is smooth enough in its hyperparameters for them to be useful.",
    importance=2, length=3)
SCIEUR = reading(
    "The Curse of Unrolling: Rate of Differentiating Through Optimization",
    "https://arxiv.org/abs/2209.13271",
    "Scieur, Bertrand, Gidel and Pedregosa (NeurIPS 2022)",
    "On quadratics, the unrolled derivative converges to the implicit (IF) one, but a fast optimizer "
    "first goes through a burn-in phase, of length set by the condition number, where the error grows.",
    importance=2, length=3)
SCHIOPPA = reading(
    "Theoretical and Practical Perspectives on what Influence Functions Do",
    "https://arxiv.org/abs/2305.16971",
    "Schioppa, Filippova, Titov and Zablotskaia (NeurIPS 2023)",
    "Which IF assumptions can be repaired and which cannot: influence fades over training time even "
    "with deterministic training, so an IF predicts a limited amount of fine-tuning on the perturbed "
    "loss, not retraining from scratch.",
    importance=2, length=3)

KREER = reading(
    "Bayesian Influence Functions for Hessian-Free Data Attribution",
    "https://arxiv.org/abs/2509.26544",
    "Kreer, Wu, Adam, Furman and Hoogland (ICLR 2026)",
    "The BIF. Read §2.2–2.3 (definition as a covariance, the localized posterior), §3.1 (SGLD "
    "estimation; their practical settings are in Appendix B) and §4.2 (LDS on CIFAR-10, roughly "
    "matching EK-FAC). Appendix A is the expansion of Walkthrough 3B.",
    importance=5, length=3)
MLODOZENIEC = reading(
    "Distributional Training Data Attribution: What do Influence Functions Sample?",
    "https://arxiv.org/abs/2506.12965",
    "Mlodozeniec et al. (2025)",
    "Why IFs work for *distributions* of trained models. §4.1, Theorem 2: unrolled SGD responses converge "
    "to the IF (Stop 2, [Walkthrough 2C](#da2-unroll-c)). §4.2, Theorem 3: the IF is the KL-optimal first-order shift of a "
    "low-temperature Gibbs distribution ([Walkthrough 3C](#da2-bif-c) below). The paper writes the mean loss, "
    "so its response is $n$ times ours.",
    importance=4, length=3)
BAKER = reading(
    "Structural Inference: Interpreting Small Language Models with Susceptibilities",
    "https://arxiv.org/abs/2504.18274",
    "Baker, Wang, Hoogland and Murfet (ICLR 2026)",
    "Susceptibilities: the response of a posterior expectation to a shift of the data distribution, again "
    "a covariance. The BIF is the case where the shift upweights one example; the paper uses a matrix of "
    "them to find modules in a small transformer.",
    importance=2, length=3)

LEE = reading(
    "Influence Dynamics and Stagewise Data Attribution",
    "https://arxiv.org/abs/2510.12071",
    "Lee, Smith, Adam and Hoogland (ICLR 2026)",
    "Influence as a time series, measured with the BIF at each checkpoint. Read §2.3 (why stagewise "
    "learning predicts sign flips), §3 with Figs. 3–4 (a hierarchy of 8 items in a deep linear "
    "network: dog helps sparrow early and harms it later) and §4 with Fig. 5 (Pythia-14M: the influence "
    "between left and right delimiters flips sign during training). For the toy model's derivation, "
    "use [Walkthrough 4](#da2-dln).",
    importance=4, length=3)

TRAK = reading(
    "TRAK: Attributing Model Behavior at Scale",
    "https://arxiv.org/abs/2303.14186",
    "Park, Georgiev, Ilyas, Leclerc and Mądry (2023)",
    "Read §3 (the five steps: linearize, randomly project the gradients, a one-step Newton estimate in "
    "the projected space, ensemble over models, sparsify) and §4 (LDS against other methods).",
    importance=4, length=3)
DATAMODELS = reading(
    "Datamodels: Predicting Predictions from Training Data",
    "https://arxiv.org/abs/2202.00622",
    "Ilyas, Park, Engstrom, Leclerc and Mądry (2022)",
    "Definition 1 and §2: a (linear) model from the subset indicator to an output, fit by LASSO on "
    "hundreds of thousands of retrainings; the target that the LDS measures.",
    importance=3, length=3)

LESS = reading(
    "LESS: Selecting Influential Data for Targeted Instruction Tuning",
    "https://arxiv.org/abs/2402.04333",
    "Xia, Malladi, Gururangan, Arora and Chen (ICML 2024)",
    "Attribution as data selection: score instruction-tuning examples by an Adam-aware gradient "
    "similarity to a few target examples (projected LoRA gradients, summed over checkpoints); "
    "training on the top 5% often beats the full set.",
    importance=2, length=3)
SURVEY = reading(
    "Training Data Influence Analysis and Estimation: A Survey",
    "https://arxiv.org/abs/2212.04612",
    "Hammoudeh and Lowd (journal Machine Learning)",
    "A map of the field up to 2022, organized by estimator; useful for finding older methods.",
    importance=2, length=4)

EXTRAS = "\n\n**Claude-suggested extras**\n"

# ======================================================================= cells
CELLS = []

CELLS += [
    markdown(
        r"""
# DA2 — Methods: an ordered tour

**Data Attribution, Iliad Intensive 2026.** Adapted from [Louis Jaburi's Iliad notes](https://iliad-intensive.org/learning/data-attribution/) ([CC BY 4.0](https://iliad-intensive.org/license/)), with changes.

DA1 built one formula, the damped Gauss–Newton influence function, and asked
which question it answers. This page tours the methods that answer the
question differently, or answer it at scale. **Read the capsule at the top of
each stop, in order** (about 30 minutes for all six). Then go deep where you
like. Each stop names one or two papers and the sections worth reading, and
most offer a **walkthrough**: our own annotated derivation, an alternative to
working through the paper's.

**Conventions** (as in [DA1 §2](da1_core.html#da1-2)):
- The training objective is $L(w;\beta)=\sum_i\beta_iL_i(w)$, with $L_i$ the loss on example $z_i$; $\beta=\mathbf 1$ is the actual run.
- A method says what training produces as a function of the data weights, $\beta\mapsto\pi(\beta)$ (a stationary point, the last iterate, a posterior). The **influence** of $z_i$ on a scalar observable $\phi$ is $\mathcal I(z_i,\phi)=\partial\phi(\pi(\beta))/\partial\beta_i$ at $\beta=\mathbf 1$.
- Deleting $z_i$ is $\beta_i:1\to0$, so to first order $\Delta\phi\approx-\mathcal I$.
- An upweighting that lowers a query loss is helpful and has **negative** influence on that loss.
- SGD written on a mean loss shows its $1/n$ (full batch) or $1/b$ (minibatch) explicitly.
- The LDS (DA1 §5) is the rank correlation, over random training subsets, between predicted and actual $\phi$.
- $H$ and $G$ are the Hessian and Gauss–Newton matrix of the summed loss. The damped IF is $\mathcal I_\lambda(z_i,\phi)=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$.

**One lens.** In their local or frozen-curvature approximations, several of
these methods share a structure: a curvature matrix, a regularization of its
small eigenvalues, a training gradient and a query gradient. But they answer
different questions, and the damping-like quantity means something different
in each. [DA1 §6](da1_core.html#da1-6) tabulates the spectral filters with their
assumptions. The exact BIF, general unrolling and datamodels have no such form.

**The tour**, with what each method takes $\pi(\beta)$ to be.
1. [Influence functions at scale](#da2-stop-1): EK-FAC makes the damped Gauss–Newton IF computable for language models. $\pi$: a stationary point.
2. [Unrolling](#da2-stop-2): differentiate the actual training run, including its order and length. $\pi$: the last iterate.
3. [Bayesian influence functions](#da2-stop-3): a covariance under a local posterior, with no Hessian inverse. $\pi$: a local posterior.
4. [Influence changes over training](#da2-stop-4): attribution is a time series, and can change sign. $\pi$: the iterate at each time $t$.
5. [Datamodels and TRAK](#da2-stop-5): fit the map from training subsets to outputs, or a linearized estimate of it. $\pi$: retraining from scratch on a subset.
6. [Open problems and uses](#da2-stop-6).

The code cells (numpy and torch on CPU) run in well under a minute in total.
The setup cell defines plotting colours and a rank correlation.
"""
    ),
    code(
        r"""
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # fixed series order
GREY = "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2, "font.size": 10})


def rank_corr(x, y):
    # Spearman rank correlation (no ties expected in continuous scores)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]
"""
    ),
]

# ======================================================================= stop 1
CELLS += [
    markdown(
        r"""
<a id="da2-stop-1"></a>

## 1. Influence functions at scale

**Question.** How do we compute the damped Gauss–Newton IF,
$\mathcal I_\lambda(z_i,\phi)=-\nabla\phi^\top(G+\lambda I)^{-1}\nabla L_i$, when $w$
has billions of entries? Here $G$ is the Gauss–Newton matrix of the summed
training loss, and $\pi(\beta)$ is still the stationary point of [DA1 §4](da1_core.html#da1-4)'s
proximal objective (the PBRF, proximal Bregman response function). The target is unchanged; only the estimator is new.

**How.** Never form $G$ or its inverse. Keep one block of $G$ per layer, and
approximate each block by a Kronecker product $S\otimes A$ of two small
matrices: the second moments of the gradients at the layer's outputs and of
its inputs. A Kronecker product is inverted through the eigendecompositions of
its two factors. EK-FAC keeps that eigenbasis but refits the eigenvalues to the
data. Then each query costs one solve $v=(G+\lambda I)^{-1}\nabla\phi$, and each
training example one gradient and one dot product, $-v^\top\nabla L_i$.

**What it bought.** Grosse et al. computed IFs for language models up to 52B
parameters. For a model's answer about being shut down, the most influential
training sequences of the largest model are about self-preservation, related
to the query at a high level of abstraction rather than by shared words. The
distribution of influence is heavy-tailed, yet spread over many sequences.
**Caveat:** each approximation (dropping cross-layer blocks, the Kronecker
factorization, the eigenvalue refit, the choice of $\lambda$) changes the
answer, and its agreement with the exact damped Gauss–Newton IF has to be
measured. [Code 1](#da2-ekfac-code) measures it on a toy network.
"""
        + GROSSE
        + "\n\n**Also:**\n" + KFAC + EKFAC + EXTRAS + LOGRA
        + r"""
<a id="da2-ekfac"></a>

### Walkthrough 1: EK-FAC for one linear layer

*Substitutes for* Grosse et al. §2.2.2–2.2.3 and §3.1–3.2 (the method), not
their findings (§5). *Shared objective:* say which matrix EK-FAC approximates, which
independence assumption it makes, and how a damped inverse is applied without
ever being formed.

**Notation.** One linear layer $s=Wa$ inside a network: input $a\in\mathbb R^d$
(append a 1 for a bias), weights $W\in\mathbb R^{m\times d}$, output
(pre-activation) $s\in\mathbb R^m$. On input $x$ with label $y$ the loss is
$\ell(f(x),y)$, and $g=\partial\ell/\partial s\in\mathbb R^m$ is the gradient
at the layer's output. The per-example weight gradient is $\nabla_W\ell=g\,a^\top$.
Flatten matrices row by row, so $\mathrm{vec}(g\,a^\top)=g\otimes a$. Write
$\mathbb E_x$ for the average over the $n$ training inputs. The Gauss–Newton
matrix of the summed loss is $G=n\bar G$ with $\bar G$ an average; damping
$\lambda$ on $G$ is damping $\lambda/n$ on $\bar G$. We work with $\bar G$.

**Step 1. The curvature target.** IFs at scale use $G$ rather than $H$ (DA1 §4),
and $G$ must be built from gradients, the one thing backprop gives cheaply. The
right gradients use labels *sampled from the model*, not the observed labels.

*Exercise.* For a classifier with logits $z=f(x)$, probabilities
$p=\mathrm{softmax}(z)$ and $\ell=-\log p_y$:
1. Show that for a label $\hat y$ drawn from $p$,
   $\mathbb E_{\hat y\sim p}\big[\nabla_z\ell(z,\hat y)\nabla_z\ell(z,\hat y)^\top\big]=\nabla_z^2\ell=\mathrm{diag}(p)-pp^\top$.
2. Conclude that the layer's block of $\bar G$ is
   $\bar G_{WW}=\mathbb E_x\mathbb E_{\hat y}\big[(g_{\hat y}g_{\hat y}^\top)\otimes(aa^\top)\big]$,
   where $g_{\hat y}$ is the output gradient computed with label $\hat y$.
3. What matrix do the observed labels give instead, and how does it compare
   with $\bar G$ near the end of training?
"""
    ),
    answer(
        r"""
**1.** $\ell(z,\hat y)=-z_{\hat y}+\log\sum_ce^{z_c}$, so $\nabla_z\ell=p-e_{\hat y}$, and
$\nabla^2_z\ell=\mathrm{diag}(p)-pp^\top$, independent of the label. Averaging over $\hat y\sim p$,

$$
\mathbb E\big[(p-e_{\hat y})(p-e_{\hat y})^\top\big]=\sum_cp_c\,(e_c-p)(e_c-p)^\top
=\mathrm{diag}(p)-pp^\top ,
$$

using $\sum_cp_c=1$ to expand.

**2.** With $J=\partial z/\partial w$, the Gauss–Newton contribution of one input is
$J^\top\nabla_z^2\ell\,J=\mathbb E_{\hat y}[J^\top\nabla_z\ell\,\nabla_z\ell^\top J]
=\mathbb E_{\hat y}[\nabla_w\ell(\hat y)\nabla_w\ell(\hat y)^\top]$. Restricted to the
layer, $\nabla_W\ell=g_{\hat y}a^\top$, i.e. $g_{\hat y}\otimes a$ after flattening, and the
mixed-product rule gives $(g\otimes a)(g\otimes a)^\top=(gg^\top)\otimes(aa^\top)$. Average over $x$.
The input $a$ does not depend on $\hat y$, so only $g$ is averaged over labels.

**3.** The observed labels give the *empirical Fisher*
$F_{\rm emp}=\mathbb E_x[\nabla\ell(y)\nabla\ell(y)^\top]$. Near interpolation $p\approx e_y$,
so $p-e_y=O(\varepsilon)$ with $\varepsilon=1-p_y$: $F_{\rm emp}=O(\varepsilon^2)$ while
$\bar G=O(\varepsilon)$. The two shrink at different rates, and their spectra differ ([TD §2.6](../td/td2_6_curvature.html#td26-1)
has the details). Grosse et al. and Martens and Grosse sample $\hat y$; one sample per
input gives an unbiased estimate of $\bar G$.
"""
    ),
    markdown(
        r"""
**Step 2. The Kronecker factorization.** The block $\bar G_{WW}$ is
$md\times md$, and for a $4096\times4096$ layer $md\approx1.7\times10^7$. K-FAC replaces the average of a product by the product of averages,
which needs only an $m\times m$ and a $d\times d$ matrix. That is exact only if
input and output gradient are uncorrelated in a specific sense.

*Exercise.* Let $S=\mathbb E_x\mathbb E_{\hat y}[gg^\top]$ and $A=\mathbb E_x[aa^\top]$. K-FAC sets
$\bar G_{WW}\approx S\otimes A$.
1. Show this is exact when $\mathbb E[g_ig_j\,a_ka_l]=\mathbb E[g_ig_j]\,\mathbb E[a_ka_l]$ for all
   indices, for example when $g$ and $a$ are independent.
2. Compare the storage of $\bar G_{WW}$ and of $(S,A)$.
"""
    ),
    answer(
        r"""
**1.** Entry $((i,k),(j,l))$ of $\bar G_{WW}$ is $\mathbb E[g_ig_j\,a_ka_l]$; the same entry of
$S\otimes A$ is $S_{ij}A_{kl}=\mathbb E[g_ig_j]\,\mathbb E[a_ka_l]$. They agree for all indices
exactly when the fourth moments factorize. In a real network they do not in general:
$g$ depends on the input through the network's prediction, and so does $a$. The code at the
end of this stop measures the error.

**2.** $\bar G_{WW}$ has $md(md+1)/2$ independent entries; $S$ and $A$ have
$m(m+1)/2+d(d+1)/2$. For $m=d=4096$: about $1.4\times10^{14}$ against $1.7\times10^7$.
"""
    ),
    markdown(
        r"""
**Step 3. A damped inverse without forming it.** We need
$v=(\bar G_{WW}+\bar\lambda I)^{-1}u$ for a query gradient $u$, once per query. In this
step $\bar\lambda$ is the damping on $\bar G$, so $\lambda=n\bar\lambda$ on $G$. The
eigendecomposition of $S\otimes A$ comes from those of $S$ and $A$, so the
inverse costs two small eigendecompositions and four matrix products. The
damping goes on the *joint* eigenvalues.

*Exercise.* Let $S=Q_SD_SQ_S^\top$ and $A=Q_AD_AQ_A^\top$, with $D_S=\mathrm{diag}(\sigma)$ and
$D_A=\mathrm{diag}(\alpha)$ holding the eigenvalues.
1. Show $S\otimes A=(Q_S\otimes Q_A)(D_S\otimes D_A)(Q_S\otimes Q_A)^\top$, so its
   eigenvalues are the products $\sigma_i\alpha_j$.
2. For $u=\mathrm{vec}(U)$ with $U\in\mathbb R^{m\times d}$, show
   $$(S\otimes A+\bar\lambda I)^{-1}u=\mathrm{vec}\Big(Q_S\big[(Q_S^\top UQ_A)\oslash(\sigma\alpha^\top+\bar\lambda)\big]Q_A^\top\Big),$$
   where $\oslash$ divides entrywise and $(\sigma\alpha^\top+\bar\lambda)_{ij}=\sigma_i\alpha_j+\bar\lambda$.
   Count the cost.
3. K-FAC *optimizers* often damp each factor instead, using
   $(S+\sqrt{\bar\lambda} I)\otimes(A+\sqrt{\bar\lambda} I)$. Why is that a different operator, and
   which directions does it damp more?
"""
    ),
    answer(
        r"""
**1.** The mixed-product rule $(B\otimes C)(D\otimes E)=BD\otimes CE$ gives
$(Q_S\otimes Q_A)(D_S\otimes D_A)(Q_S\otimes Q_A)^\top=(Q_SD_SQ_S^\top)\otimes(Q_AD_AQ_A^\top)$.
$Q_S\otimes Q_A$ is orthogonal, and $D_S\otimes D_A$ is diagonal with entries $\sigma_i\alpha_j$.

**2.** With row-major flattening, $(B\otimes C)\,\mathrm{vec}(U)=\mathrm{vec}(BUC^\top)$ (check it on
$U=ga^\top$: $(Bg)\otimes(Ca)=\mathrm{vec}(Bg\,a^\top C^\top)$). So
$(Q_S\otimes Q_A)^\top u=\mathrm{vec}(Q_S^\top UQ_A)$: the coordinates of $u$ in the eigenbasis.
Dividing coordinate $(i,j)$ by $\sigma_i\alpha_j+\bar\lambda$ applies the inverse of the damped
diagonal, and $Q_S\otimes Q_A$ maps back. Cost: $O(m^3+d^3)$ once per layer for the
eigendecompositions, then $O(md(m+d))$ per vector, against $O(m^3d^3)$ for a dense solve.

**3.** Its eigenvalues are $(\sigma_i+\sqrt{\bar\lambda})(\alpha_j+\sqrt{\bar\lambda})=\sigma_i\alpha_j+\bar\lambda+\sqrt{\bar\lambda}(\sigma_i+\alpha_j)$.
The extra $\sqrt{\bar\lambda}(\sigma_i+\alpha_j)$ damps more wherever either factor is large,
including directions with a small product. It is not $\bar G+\bar\lambda I$ for any $\bar G$ of
the form $S\otimes A$, so the IF it gives is not the damped Gauss–Newton IF. Grosse et al.
add the damping to the joint eigenvalues (their Eq. 21).
"""
    ),
    markdown(
        r"""
**Step 4. The eigenvalue refit (EK-FAC).** The Kronecker eigenbasis can be good
even when the Kronecker eigenvalues $\sigma_i\alpha_j$ are not. EK-FAC keeps the
basis $Q=Q_S\otimes Q_A$ and refits the eigenvalues as the second moments of the
actual gradients in that basis. That is the best diagonal in the basis, and it
removes the independence assumption along the diagonal.

*Exercise.* Here $g$ is computed with $\hat y\sim p$, as in Step 1.
1. Show that the diagonal $\Lambda$ minimizing $\|\bar G_{WW}-Q\,\mathrm{diag}(\Lambda)\,Q^\top\|_F$ is
   $\Lambda_{ij}=\mathbb E\big[(Q_S^\top g)_i^2\,(Q_A^\top a)_j^2\big]$. Deduce that EK-FAC is never
   further from $\bar G_{WW}$ in Frobenius norm than K-FAC.
2. Show that if $g$ and $a$ are independent, $\Lambda_{ij}=\sigma_i\alpha_j$: EK-FAC reduces to K-FAC.
3. Write the EK-FAC damped IF for one layer, in summed-loss units.
"""
    ),
    answer(
        r"""
**1.** The Frobenius norm is invariant under orthogonal conjugation:
$\|\bar G-Q\Lambda Q^\top\|_F=\|Q^\top\bar GQ-\Lambda\|_F$. Over diagonal $\Lambda$ this is minimized by
copying the diagonal of $Q^\top\bar GQ$; the off-diagonal part is the irreducible error. The
basis vector for index $(i,j)$ is $q_{ij}=(Q_S)_{:i}\otimes(Q_A)_{:j}$, and
$q_{ij}^\top(g\otimes a)=(Q_S^\top g)_i\,(Q_A^\top a)_j$, so
$\Lambda_{ij}=q_{ij}^\top\bar Gq_{ij}=\mathbb E[(Q_S^\top g)_i^2(Q_A^\top a)_j^2]$. K-FAC is another
diagonal in the same basis, so its error is at least as large.

**2.** If $g$ and $a$ are independent the expectation factorizes:
$\mathbb E[(Q_S^\top g)_i^2]\,\mathbb E[(Q_A^\top a)_j^2]=(Q_S^\top SQ_S)_{ii}(Q_A^\top AQ_A)_{jj}=\sigma_i\alpha_j$.

**3.** For the summed loss the curvature is $n\bar G$, with eigenvalues $n\Lambda_{ij}$ in the
EK-FAC basis. With $U_\phi$ and $U_i$ the layer blocks of $\nabla\phi$ and $\nabla L_i$ (as $m\times d$
matrices),

$$
\mathcal I_\lambda(z_i,\phi)\approx-\Big\langle Q_S\big[(Q_S^\top U_\phi Q_A)\oslash(n\Lambda+\lambda)\big]Q_A^\top,\;U_i\Big\rangle ,
$$

summed over layers (the cross-layer blocks of $G$ are dropped). The bracket is computed once
per query; each training example then costs one gradient and one inner product.
"""
    ),
    markdown(
        r"""
**From one layer to a language model.** Grosse et al. apply this to the MLP
layers only. Each training sequence contributes one pseudo-gradient per token
position: they fit $S$ and $A$ as if tokens were independent, then refit
$\Lambda$ on the gradients summed over the sequence (§3.1). Two engineering
steps make the search feasible. *TF-IDF filtering* keeps about $10^4$ candidate
sequences per query, which they note biases the results. *Query batching*
stores each query's preconditioned gradient $Q_S[\cdots]Q_A^\top$ as a rank-32
matrix per layer, so hundreds of queries share one pass over the training
gradients (§3.2).

> **Scope.** We derived the one-layer algebra of K-FAC and EK-FAC and the optimality of the refit. Not derived: dropping the cross-layer blocks, the token-independence fit for sequences, and the size of any approximation error, which is empirical.

<a id="da2-ekfac-code"></a>

### Code 1: what each approximation costs

A two-layer tanh network (147 parameters) is trained briefly on three Gaussian
classes ($n=240$; 200 Adam steps, training loss about 0.05). We build four curvature matrices: the exact Gauss–Newton $\bar G$
(averaged over model-sampled labels exactly, not by sampling), its
block-diagonal part, K-FAC and EK-FAC. We then compute damped IFs of every
training example on the test losses of 12 held-out points, **with the same
$\lambda$** for all four. The block-diagonal column isolates block omission;
K-FAC and EK-FAC against block-diagonal isolate the Kronecker error. Look for:
- EK-FAC's Frobenius error is below K-FAC's in every block, as Step 4 guarantees;
- the Kronecker step costs far more than dropping the cross-layer blocks: rank correlation with the exact IF is about 0.65–0.75 for block-diagonal and 0.3–0.45 for K-FAC and EK-FAC;
- EK-FAC's better curvature fit raises the Pearson correlation a little and leaves the rank correlation flat;
- larger damping raises the Pearson correlations, because it hides the small-eigenvalue directions where the approximations differ most.
"""
    ),
    code(
        r"""
t0 = time.time()
g = torch.Generator().manual_seed(0)
C, din, h, n_per = 3, 8, 12, 80
mu = torch.randn(C, din, generator=g)
X = torch.cat([mu[c] + torch.randn(n_per, din, generator=g) for c in range(C)])
y = torch.arange(C).repeat_interleave(n_per)
Xq = torch.cat([mu[c] + torch.randn(4, din, generator=g) for c in range(C)])
yq = torch.arange(C).repeat_interleave(4)
n = len(X)
shapes = [(h, din + 1), (C, h + 1)]
sizes = [h * (din + 1), C * (h + 1)]


def forward(th, X):
    # two layers, each s = W a with a bias column appended to a
    W1, W2 = (p.reshape(s) for p, s in zip(torch.split(th, sizes), shapes))
    a1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    hid = torch.tanh(a1 @ W1.T)
    a2 = torch.cat([hid, torch.ones(len(X), 1)], 1)
    return a2 @ W2.T, a1, a2, hid


th = (0.5 * torch.randn(sum(sizes), generator=g)).requires_grad_(True)
opt = torch.optim.Adam([th], lr=0.02)
for _ in range(200):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(forward(th, X)[0], y)
    loss.backward()
    opt.step()
th = th.detach()
P = th.numel()
print(f"P = {P} parameters, train loss {loss.item():.4f}")

# exact Gauss-Newton (mean over inputs, exact average over model-sampled labels)
z, a1, a2, hid = forward(th, X)
p = torch.softmax(z, 1)
Hz = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]          # E_yhat[(p-e)(p-e)^T], (n, C, C)
J = torch.func.vmap(torch.func.jacrev(lambda t, x: forward(t, x[None])[0][0]), in_dims=(None, 0))(th, X)
G = torch.einsum("ncp,ncd,ndq->pq", J, Hz, J) / n
blocks = [slice(0, sizes[0]), slice(sizes[0], P)]
G_bd, G_kfac, G_ekfac = (torch.zeros_like(G) for _ in range(3))
# E_yhat[g g^T] at each layer's output, per input
W2 = th[sizes[0]:].reshape(shapes[1])[:, :h]
Sg2 = Hz
Sg1 = torch.einsum("nh,ch,ncd,dk,nk->nhk", 1 - hid**2, W2, Hz, W2, 1 - hid**2)
for blk, Sg, a in [(blocks[0], Sg1, a1), (blocks[1], Sg2, a2)]:
    G_bd[blk, blk] = G[blk, blk]
    S, A = Sg.mean(0), a.T @ a / n
    G_kfac[blk, blk] = torch.kron(S, A)                               # row-major vec(g a^T) = g (x) a
    _, QS = torch.linalg.eigh(S)
    _, QA = torch.linalg.eigh(A)
    # EK-FAC: Lambda_ij = E_x[ E_yhat[(QS^T g)_i^2] (QA^T a)_j^2 ]
    Lam = torch.einsum("ni,nj->ij", torch.einsum("ia,nij,ja->na", QS, Sg, QS), (a @ QA) ** 2) / n
    Q = torch.kron(QS, QA)
    G_ekfac[blk, blk] = Q @ torch.diag(Lam.flatten()) @ Q.T
    fro = lambda M: torch.linalg.norm(M - G[blk, blk]).item() / torch.linalg.norm(G[blk, blk]).item()
    print(f"layer block of size {blk.stop - blk.start}: relative Frobenius error  "
          f"K-FAC {fro(G_kfac[blk, blk]):.2f}   EK-FAC {fro(G_ekfac[blk, blk]):.2f}")

grad_loss = torch.func.vmap(torch.func.grad(
    lambda t, x, yy: torch.nn.functional.cross_entropy(forward(t, x[None])[0], yy[None])), in_dims=(None, 0, 0))
g_train, g_query = grad_loss(th, X, y), grad_loss(th, Xq, yq)
mean_eig = torch.linalg.eigvalsh(n * G).mean().item()
names = ["exact GN", "block-diag GN", "K-FAC", "EK-FAC"]
print("\nmean over 12 queries of the correlation with the exact damped GN IF (Pearson r / Spearman rho)")
print(f"{'lambda':>22s}" + "".join(f"{nm:>18s}" for nm in names[1:]))
for rel in [1e-3, 1e-2, 1e-1]:
    lam = rel * mean_eig
    IF = {nm: (-(g_query @ torch.linalg.solve(n * M + lam * torch.eye(P), g_train.T))).numpy()
          for nm, M in zip(names, [G, G_bd, G_kfac, G_ekfac])}
    row = f"{rel:g} x mean eig(nG) = {lam:.1e}"
    for nm in names[1:]:
        r = np.mean([np.corrcoef(IF[nm][k], IF["exact GN"][k])[0, 1] for k in range(len(Xq))])
        rho = np.mean([rank_corr(IF[nm][k], IF["exact GN"][k]) for k in range(len(Xq))])
        row += f"{r:>11.2f} / {rho:.2f}"
    print(row)
    if rel == 1e-1:
        IF_show = IF

fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True, sharey=True)
k = 0
for ax, nm, c in zip(axs, names[1:], COL):
    ax.scatter(IF_show["exact GN"][k], IF_show[nm][k], s=10, color=c)
    lim = np.abs(IF_show["exact GN"][k]).max() * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1)
    ax.set_title(nm)
    ax.set_xlabel("exact damped GN IF")
axs[0].set_ylabel("approximate IF")
fig.suptitle(r"IF of each training example on one test loss, $\lambda=0.1\times$ mean eigenvalue", fontsize=10)
fig.tight_layout()
plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
]

# ======================================================================= stop 2
CELLS += [
    markdown(
        r"""
<a id="da2-stop-2"></a>

## 2. Unrolling: differentiate the actual training run

**Question.** If example $z_i$ had been weighted slightly more in *this*
training run (same initialization, same data order, same optimizer), how would
$\phi(w_T)$ change? Here $\pi(\beta)=w_T(\beta)$, the last iterate as a function
of the data weights. Order, optimizer and training length become part of the
answer: an example seen in the first minibatch and one seen in the last can
have very different effects, which no endpoint formula can distinguish.

**How.** Training is a composition of steps, so its derivative is a product of
step Jacobians, exactly as backprop differentiates a composition of layers
([Walkthrough 2A](#da2-unroll-a)). Each step that uses $z_i$ injects its
gradient, and the later steps propagate it. *TracIn* keeps only the
injections, each paired with the query gradient at the same step: it
allocates the decrease of the test loss along the trajectory to the examples
that were used, a different estimand from the endpoint sensitivity. *SOURCE*
freezes the curvature within a few training segments, which turns unrolling
into an IF with a filter $F(s)$ in place of $1/s$ on each curvature eigenvalue
$s$ ([Walkthrough 2B](#da2-unroll-b); DA1 §6's table). *MAGIC* computes the
exact derivative by reverse mode through training.

**Main result and caveat.** For frozen positive-definite curvature, long
training makes the unrolled response forget its history and converge to the IF
([Walkthrough 2C](#da2-unroll-c)). An exact
derivative is exact only for infinitesimal reweighting of this run: MAGIC's
near-perfect LDS holds when about 1% of the data is dropped with the
randomness fixed, and degrades for larger deletions. Even in the ridge
regression of the code below, deleting one example moves $\phi$ 30% more than
the first-order prediction.
"""
        + SOURCE + MAGIC
        + "\n\n**Also:**\n" + TRACIN + ENGSTROM + EXTRAS + SCIEUR + SCHIOPPA
        + r"""
<a id="da2-unroll-a"></a>

### Walkthrough 2A: the unrolling formula

*Substitutes for* SOURCE §3.1 and the set-up of MAGIC §2. *Shared objective:*
derive the influence of an example on the last iterate as a sum over the steps
that used it, and know what changes for adaptive optimizers.

Unrolling is to a training run what backprop is to a forward pass:

| | Forward pass | Training run |
|---|---|---|
| inputs | weights $w$ | data weights $\beta$ |
| intermediate states | activations $a_1,\dots,a_L$ | iterates $w_1,\dots,w_T$ |
| output | loss of the logits | observable $\phi(w_T)$ |
| differentiation | backprop | unrolling |

**Notation.** Minibatch SGD on the mean minibatch loss, with weights $\beta$:
$$
w_{t+1}=w_t-\frac{\eta_t}{b}\sum_{j\in B_t}\beta_j\nabla L_j(w_t),
$$
with fixed $w_0$ and a fixed batch sequence $B_0,\dots,B_{T-1}$ of size $b$. Let
$\hat H_t=\frac1b\sum_{j\in B_t}\nabla^2L_j(w_t)$ be the minibatch Hessian,
$J_t=I-\eta_t\hat H_t$ the step Jacobian, and $J_{(t+1):T}=J_{T-1}\cdots J_{t+1}$ the
ordered product ($I$ when $t=T-1$). All derivatives are at $\beta=\mathbf 1$.

**Step A1. One step.** $\beta_i$ enters a step twice: directly, through its
weight in the sum, and indirectly, through $w_t$, which already depends on
$\beta_i$.

*Exercise.* Show that
$\dfrac{\partial w_{t+1}}{\partial\beta_i}=J_t\dfrac{\partial w_t}{\partial\beta_i}-\dfrac{\eta_t}{b}\,\mathbf 1[i\in B_t]\,\nabla L_i(w_t)$,
with $\partial w_0/\partial\beta_i=0$.
"""
    ),
    answer(
        r"""
Differentiate the update with the chain rule. The sum depends on $\beta_i$ through the
coefficient $\beta_i$ (only if $i\in B_t$) and through $w_t$ in every term:

$$
\frac{\partial w_{t+1}}{\partial\beta_i}=\frac{\partial w_t}{\partial\beta_i}
-\frac{\eta_t}{b}\sum_{j\in B_t}\nabla^2L_j(w_t)\,\frac{\partial w_t}{\partial\beta_i}
-\frac{\eta_t}{b}\,\mathbf 1[i\in B_t]\,\nabla L_i(w_t),
$$

using $\beta_j=1$. The first two terms combine to $(I-\eta_t\hat H_t)\,\partial w_t/\partial\beta_i=J_t\,\partial w_t/\partial\beta_i$.
The initial point does not depend on the data weights.
"""
    ),
    markdown(
        r"""
**Step A2. Unroll.** Solving the recursion is substituting it into itself,
as backprop does layer by layer.

*Exercise.* Show
$$
\frac{\partial w_T}{\partial\beta_i}=-\sum_{t=0}^{T-1}\frac{\eta_t}{b}\,\mathbf 1[i\in B_t]\,J_{(t+1):T}\,\nabla L_i(w_t),\qquad
\mathcal I(z_i,\phi)=\nabla\phi(w_T)^\top\frac{\partial w_T}{\partial\beta_i}.
$$
What do you get if you set every $J=I$ and replace $\nabla\phi(w_T)$ by $\nabla\phi(w_t)$?
"""
    ),
    answer(
        r"""
Write $r_t=\partial w_t/\partial\beta_i$ and $c_t=-\frac{\eta_t}{b}\mathbf 1[i\in B_t]\nabla L_i(w_t)$, so
$r_{t+1}=J_tr_t+c_t$ and $r_0=0$. Then $r_1=c_0$, $r_2=J_1c_0+c_1$, and by induction
$r_T=\sum_{t=0}^{T-1}J_{T-1}\cdots J_{t+1}\,c_t=\sum_tJ_{(t+1):T}c_t$. The chain rule gives
$\mathcal I=\nabla\phi(w_T)^\top r_T$. Each term is one appearance of $z_i$: its gradient at that
step, carried forward by all later steps.

With every $J=I$ and the query gradient taken at the same step,
$-\frac1b\sum_{t:\,i\in B_t}\eta_t\nabla\phi(w_t)^\top\nabla L_i(w_t)$. For $\phi$ a test loss this
is minus TracIn's minibatch score (TracIn counts a loss decrease as positive). To first
order in $\eta_t$, the step $t$ changes $\phi$ by $-\eta_t\nabla\phi(w_t)^\top\hat g_t$, and the part
of $\hat g_t$ due to $z_i$ is $\nabla L_i(w_t)/b$. So TracIn allocates the decrease of $\phi$ along
the trajectory to the examples used at each step. It does not ask how the *endpoint* would
move under reweighting, which is what the Jacobians propagate.
"""
    ),
    markdown(
        r"""
**Step A3. Adaptive optimizers.** Adam and natural-gradient methods multiply
the gradient by a matrix that itself depends on the trajectory. Treating that
matrix as fixed is the *frozen-preconditioner approximation*.

*Exercise.* Consider $w_{t+1}=w_t-\frac{\eta_t}{b}P(w_t)\sum_{j\in B_t}\beta_j\nabla L_j(w_t)$,
with $P(w)\succ0$ depending smoothly on $w$, and write $\hat g_t=\frac1b\sum_{j\in B_t}\nabla L_j(w_t)$.
1. Show that the step Jacobian acts as $v\mapsto v-\eta_t\big[P(w_t)\hat H_tv+\big(DP(w_t)[v]\big)\hat g_t\big]$.
2. In the *frozen-preconditioner approximation* one drops the $DP$ term and treats
   $P_t=P(w_t)$ as fixed. Show that the unrolled formula becomes
   $-\sum_t\frac{\eta_t}{b}\mathbf 1[i\in B_t]\,J^{(P)}_{(t+1):T}P_t\nabla L_i(w_t)$ with $J^{(P)}_t=I-\eta_tP_t\hat H_t$.
3. Adam's preconditioner is built from a running average $v_t$ of squared minibatch
   gradients, which depends on $\beta$ too. How do you differentiate it exactly?
"""
    ),
    answer(
        r"""
**1.** The step is $w\mapsto w-\eta_tP(w)\hat g(w)$. Its derivative in direction $v$ is
$v-\eta_t\big[DP(w)[v]\,\hat g(w)+P(w)\,D\hat g(w)[v]\big]$, and $D\hat g(w)[v]=\hat H_tv$. The
$DP$ term vanishes where $\hat g_t=0$, which minibatch gradients almost never are.

**2.** With $P_t$ fixed, Step A1 goes through with $\hat H_t\to P_t\hat H_t$ in the Jacobian and
$\nabla L_i\to P_t\nabla L_i$ in the direct term; unrolling as in Step A2 gives the formula.

**3.** Make the whole optimizer state the state: $s_t=(w_t,m_t,v_t)$ with update
$s_{t+1}=h_t(s_t;\beta)$. Then
$\partial\phi(s_T)/\partial\beta=\sum_t A_{t+1}\,\partial h_t/\partial\beta$, with
$A_T=\partial\phi/\partial s_T$ and $A_t=A_{t+1}\,\partial h_t/\partial s_t$: backprop through the
training loop, over the full state. This captures that $z_i$'s gradient enters $v_t$ and
so rescales every later step for all data. MAGIC computes exactly this, replaying training
from saved checkpoints to regenerate the states.
"""
    ),
    markdown(
        r"""
> **Scope.** Steps A1–A2 are exact for SGD with the randomness held fixed; for any differentiable update the same chain rule holds over the full optimizer state (Step A3). What is expensive is the Jacobian product: SOURCE approximates it, MAGIC computes it.

<a id="da2-unroll-b"></a>

### Walkthrough 2B: the SOURCE filter

*Substitutes for* SOURCE §§3.2–3.3. *Shared objective:* see how freezing the
curvature within a training segment turns the unrolled sum into an IF with a
filter $F(s)$ in place of $1/s$, and what the crossover at $s\sim1/(\eta K)$
does and does not mean.

**Notation.** One segment of $K$ steps of minibatch SGD on the mean minibatch
loss, $w_{t+1}=w_t-\frac{\eta_t}{b}\sum_{j\in B_t}\beta_j\nabla L_j(w_t)$ ($n$ examples, batch size
$b$, batches drawn uniformly, so $\Pr[i\in B_t]=b/n$). SOURCE's *stationarity approximation*: within the segment
the minibatch Hessian, the gradient of $z_i$ and the step size are constant,
$\hat H_t\approx\bar H$, $\nabla L_i(w_t)\approx\bar g_i$, $\eta_t\approx\bar\eta$, and the indicators
$\mathbf 1[i\in B_t]$ are replaced by their mean $b/n$. $\bar H$ is a mean-loss Hessian, so
its eigenvalues $s$ do not grow with $n$.

**Step B1. The segment response.** With everything frozen, the sum over steps
is a matrix geometric series.

*Exercise.* Show that the segment's contribution to $\partial w/\partial\beta_i$ at its end is
$$
-\frac{\bar\eta}{n}\sum_{k=0}^{K-1}(I-\bar\eta\bar H)^k\,\bar g_i
=-\frac1n\big[I-(I-\bar\eta\bar H)^K\big]\bar H^{-1}\bar g_i
\approx-\frac1n\big(I-e^{-\bar\eta K\bar H}\big)\bar H^{-1}\bar g_i ,
$$
so in an eigenbasis of $\bar H$, with eigenvalue $s$, it acts on $\bar g_i$ as $-\frac1nF(s)$ with
$F(s)=\big(1-e^{-\bar\eta Ks}\big)/s$. When is the last step accurate? (In summed-loss units,
eigenvalue $ns$ and step $\bar\eta/n$, the same response is $-F$ with no $1/n$: DA1 §6's SOURCE row.)
"""
    ),
    answer(
        r"""
Within the segment, the step $t$ term of Walkthrough 2A carries
$\frac{\eta_t}{b}\mathbf 1[i\in B_t]\to\frac{\bar\eta}{b}\cdot\frac bn=\frac{\bar\eta}n$, and is propagated to the
segment's end by $K-1-k$ frozen Jacobians $(I-\bar\eta\bar H)$. Relabelling, the sum is
$-\frac{\bar\eta}n\sum_{k=0}^{K-1}M^k\bar g_i$ with $M=I-\bar\eta\bar H$. The geometric series
$\sum_{k<K}M^k=(I-M^K)(I-M)^{-1}$ and $I-M=\bar\eta\bar H$ give the middle form. In the eigenbasis,
$(1-\bar\eta s)^K=e^{K\log(1-\bar\eta s)}\approx e^{-\bar\eta Ks}$, accurate when $\bar\eta s\ll1$. The
exponent is off by about $K\bar\eta^2s^2/2$, which matters only where $e^{-\bar\eta Ks}$ is not
already negligible.
"""
    ),
    markdown(
        r"""
**Step B2. Reading the filter.** $F$ plays the role of $1/s$ in the IF. Its
behaviour for small, large and negative $s$ is what "unrolling supplies its own
damping" means, and where that phrase is loose.

*Exercise.*
1. Show $F(s)\to1/s$ for $s\gg1/(\bar\eta K)$ and $F(0)=\bar\eta K$.
2. Compare with the damped IF filter $1/(s+\lambda)$ at $\lambda=1/(\bar\eta K)$: both are $\approx\bar\eta K$
   at $s=0$ and $\approx1/s$ for large $s$. Evaluate both at the crossover $s=1/(\bar\eta K)$.
   How do they approach $1/s$?
3. What does $F$ do for a negative eigenvalue?
"""
    ),
    answer(
        r"""
**1.** For $\bar\eta Ks\gg1$, $e^{-\bar\eta Ks}\to0$ and $F\to1/s$. For $s\to0$,
$1-e^{-\bar\eta Ks}=\bar\eta Ks+O(s^2)$, so $F(0)=\bar\eta K$: a direction with no curvature moves
linearly for $K$ steps, by $\bar\eta K$ per unit force, and no further.

**2.** At $s=1/(\bar\eta K)$: $F=(1-e^{-1})\,\bar\eta K\approx0.632\,\bar\eta K$, while
$1/(s+\lambda)=\bar\eta K/2$. For large $s$, $F=1/s-e^{-\bar\eta Ks}/s$ approaches $1/s$
exponentially fast, while $1/(s+\lambda)=1/s-\lambda/s^2+\dots$ approaches it only algebraically.
So $1/(\bar\eta K)$ sets the crossover scale, as damping does, but $F$ is not a ridge penalty.

**3.** For $s<0$, $F(s)=\big(e^{\bar\eta K|s|}-1\big)/|s|>0$, growing exponentially in $K$: along
negative curvature (a saddle direction) a push grows exponentially over the segment. It
stays finite for every $s$, unlike $1/(s+\lambda)$ at $s=-\lambda$.
"""
    ),
    markdown(
        r"""
**Step B3. Several segments.** SOURCE splits training into $L$ segments (it uses
$L=3$), each with its own $\bar H_\ell$, $\bar g_{i,\ell}$, $\bar\eta_\ell$ and $K_\ell$. A segment's
response is carried through the later segments by their frozen Jacobians
$\bar S_\ell=e^{-\bar\eta_\ell K_\ell\bar H_\ell}$:
$$
\frac{\partial w_T}{\partial\beta_i}\approx-\frac1n\sum_{\ell=1}^L\Big(\prod_{\ell'=\ell+1}^{L}\bar S_{\ell'}\Big)\big(I-e^{-\bar\eta_\ell K_\ell\bar H_\ell}\big)\bar H_\ell^{-1}\bar g_{i,\ell},
$$
with later segments on the left. Once the $\bar H_\ell$ differ, this is not one
filter applied to one gradient.

> **Scope.** We derived the one-segment filter under stationarity, the exponential approximation and averaged batch membership. These hide changing gradients, ordered non-commuting Jacobians across segments, and any correlation between when $z_i$ appears and the curvature at that time.

<a id="da2-unroll-c"></a>

### Walkthrough 2C: from unrolling to the influence function

*Covers* the deterministic mechanism behind Mlodozeniec et al. §4.1 (Theorem 2);
the stochastic-approximation proof is not reproduced. *Shared objective:* prove that for frozen positive-definite curvature and
full-batch GD the unrolled response forgets its history and converges to the
IF, and see exactly what goes wrong along flat directions.

**Notation.** Full-batch GD on the mean loss, with $z_k$ downweighted:
$$
w_{t+1}=w_t-\frac\eta n\sum_j\beta_j\nabla L_j(w_t),\qquad \beta=\mathbf 1-\varepsilon e_k .
$$
So $\beta_k=1-\varepsilon$, and **deleting $z_k$ is $\varepsilon=1$**. The response is
$r_t=\partial w_t/\partial\varepsilon$ at $\varepsilon=0$, which is $-\partial w_t/\partial\beta_k$.
$\bar H_t=\frac1n\sum_j\nabla^2L_j(w_t)$ is the mean-loss Hessian; the summed-loss
Hessian is $H=n\bar H$.

**Step C1. The response recursion.** This is Walkthrough 2A with $b=n$ and the
sign of the perturbation flipped.

*Exercise.* Show $r_{t+1}=(I-\eta\bar H_t)\,r_t+\frac\eta n\nabla L_k(w_t)$, with $r_0=0$.
"""
    ),
    answer(
        r"""
By Step A1 with $b=n$ and $B_t=\{1,\dots,n\}$,
$\partial w_{t+1}/\partial\beta_k=(I-\eta\bar H_t)\,\partial w_t/\partial\beta_k-\frac\eta n\nabla L_k(w_t)$.
Since $\beta_k=1-\varepsilon$, $r_t=-\partial w_t/\partial\beta_k$; multiply by $-1$. Downweighting
pushes $w$ *against* the descent direction of $L_k$, hence the plus sign.
"""
    ),
    markdown(
        r"""
**Step C2. A converged window.** Suppose training has reached a stationary point
$w^*$ before the last $K$ steps, so that on $T-K\le t<T$ we have $w_t=w^*$,
$\bar H_t=\bar H\succ0$ and $\nabla L_k(w_t)=g_k$. The response entering the window,
$r_{T-K}$, is whatever the earlier history produced.

*Exercise.* Show
$$
r_T=(I-\eta\bar H)^K\,r_{T-K}+\frac1n\big[I-(I-\eta\bar H)^K\big]\bar H^{-1}g_k .
$$
For $0<\eta<2/\lambda_{\max}(\bar H)$, take $K\to\infty$. What does the limit not depend on,
and how does it compare with the IF?
"""
    ),
    answer(
        r"""
Iterating the frozen recursion $K$ times,
$r_T=M^Kr_{T-K}+\frac\eta n\sum_{s=0}^{K-1}M^sg_k$ with $M=I-\eta\bar H$, and
$\sum_{s<K}M^s=(I-M^K)(\eta\bar H)^{-1}$ gives the formula. If $0<\eta<2/\lambda_{\max}$, every
eigenvalue $1-\eta s$ of $M$ has modulus below 1 (here $s>0$), so $M^K\to0$ and
$$
r_\infty=\frac1n\bar H^{-1}g_k=H^{-1}g_k .
$$
The limit does not depend on $r_{T-K}$: the inherited response decays, and the run forgets how
it got to $w^*$. $H^{-1}g_k$ is minus the upweighting IF $-H^{-1}g_k$ (DA1 §3), as it should be
for a downweighting, and deletion ($\varepsilon=1$) moves $w$ by $H^{-1}g_k$ to first order. Dropping
$r_{T-K}$ from the formula is only correct once $(I-\eta\bar H)^K$ is negligible.
"""
    ),
    markdown(
        r"""
**Step C3. Flat directions.** Now let $\bar H\succeq0$ have a null space $N$, as at a
degenerate minimum, with orthogonal projector $P_0$ onto $N$.

*Exercise.* With the same frozen window, show that
$P_0r_T=P_0r_{T-K}+\frac{K\eta}nP_0g_k$, while the component in the range of $\bar H$ still
converges to $\frac1n\bar H^+g_k$. When does the response converge? Check that in linear
regression it always does.
"""
    ),
    answer(
        r"""
$P_0$ commutes with $\bar H$ and $P_0\bar H=0$, so $P_0r_{t+1}=P_0r_t+\frac\eta nP_0g_k$; summing $K$ steps
gives the linear growth. On the range of $\bar H$ the argument of Step C2 applies with $\bar H^+$.
So $r_t$ converges if and only if $P_0g_k=0$, that is $g_k\in\mathrm{Range}(\bar H)$: the
example must not push along a direction the total loss is flat in. In linear regression
$g_k=-x_k(y_k-x_k^\top w^*)$ lies in the span of the inputs, which is $\mathrm{Range}(X^\top X)$, so
there is no drift.
"""
    ),
    markdown(
        r"""
> **Scope.** Proved: full-batch GD with frozen, positive-definite curvature, including the inherited response, and the drift along flat directions. For SGD, Theorem 2 of [Mlodozeniec et al.](#da2-stop-3) (§4.1) gives the same limit $H^+g_k$ plus a null-space part that need not converge, under regularity, bounded $(w_t,r_t)$, $g_k\in\mathrm{Range}(H)$, eigen-gap and uniform-convergence assumptions (A1–A6). Convergence of $w_t$ alone is not enough: Step C3 with $\eta_t$ in place of $\eta$ drifts without bound, since $\sum_t\eta_t=\infty$.

<a id="da2-unroll-code"></a>

### Code 2: unrolled response against the IF

Ridge regression ($n=60$, $d=8$, input scales spread over a factor 30, so the
mean-loss Hessian's eigenvalues span $0.002$ to $0.7$), trained by full-batch
GD from $w_0=0$ with $\eta=1/\lambda_{\max}$. Observable: a query loss. We
downweight one high-residual example and compare, as a function of the number
of steps $K$:
- **finite differences of actual retraining** ($\varepsilon=\pm10^{-4}$, same initialization), which test the sign and the $1/n$;
- the **exact unrolled recursion** of Step C1;
- the **frozen-curvature filter**: the parameter response $\frac1n[I-(I-\eta\bar H)^K]\bar H^{-1}g_k$ with $g_k=\nabla L_k(w^*)$, read with the current query gradient $\nabla\phi(w_K)$;
- the **IF**, $\frac1n\bar H^{-1}g_k$, read with $\nabla\phi(w^*)$.

The Hessian of ridge regression is constant, so the frozen filter is wrong
early only because the training gradient $\nabla L_k(w_t)$ is not yet $g_k$. Look for:
- the response building up over about $1/(\eta s)$ steps for each eigenvalue $s$ (the slowest takes about 344 steps);
- the inherited term of Step C2: dropping it from a 200-step window gets about half of $r_T$ wrong;
- deleting the example (printed) moves $\phi$ about 30% more than the IF predicts: first order is not deletion, even here.
"""
    ),
    code(
        r"""
t0 = time.time()
rng = np.random.default_rng(0)
n, d, mu_ridge = 60, 8, 1e-3
X = rng.standard_normal((n, d)) * np.geomspace(1, 0.03, d)
w_true = rng.standard_normal(d)
y = X @ w_true + 0.3 * rng.standard_normal(n)
Hbar = X.T @ X / n + mu_ridge * np.eye(d)                 # mean-loss Hessian (constant)
evals, Q = np.linalg.eigh(Hbar)
eta = 1.0 / evals.max()
w_star = np.linalg.solve(Hbar, X.T @ y / n)
k = int(np.argmax(np.abs((y - X @ w_star) * X[:, -1])))   # pushes along the flattest input direction
xq = rng.standard_normal(d)
yq = xq @ w_true
phi = lambda w: 0.5 * (yq - xq @ w) ** 2
grad_phi = lambda w: -(yq - xq @ w) * xq
grad_Lk = lambda w: -(y[k] - X[k] @ w) * X[k]
Kmax = 10000


def train(eps):
    # full-batch GD on (1/n) sum_j beta_j L_j + ridge, with beta_k = 1 - eps; returns all iterates
    beta = np.ones(n)
    beta[k] -= eps
    w, W = np.zeros(d), np.empty((Kmax + 1, d))
    for t in range(Kmax + 1):
        W[t] = w
        w = w - eta * (X.T @ (beta * (X @ w - y)) / n + mu_ridge * w)
    return W


W0, Wp, Wm = train(0.0), train(1e-4), train(-1e-4)
fd = (np.array([phi(w) for w in Wp]) - np.array([phi(w) for w in Wm])) / 2e-4
r, rec, R = np.zeros(d), np.empty(Kmax + 1), np.empty((Kmax + 1, d))
for t in range(Kmax + 1):                                 # Step C1, exact for this model
    R[t], rec[t] = r, grad_phi(W0[t]) @ r
    r = (np.eye(d) - eta * Hbar) @ r + eta / n * grad_Lk(W0[t])
IF = grad_phi(w_star) @ np.linalg.solve(Hbar, grad_Lk(w_star)) / n
Ks = np.unique(np.geomspace(1, Kmax, 60).astype(int))
frozen = [grad_phi(W0[K]) @ Q @ ((1 - (1 - eta * evals) ** K) / evals * (Q.T @ grad_Lk(w_star))) / n for K in Ks]
beta = np.ones(n)
beta[k] = 0
w_del = np.linalg.solve((X.T * beta) @ X / n + mu_ridge * np.eye(d), X.T @ (beta * y) / n)
print(f"eigenvalues of Hbar: {evals.min():.4f} ... {evals.max():.3f};  1/(eta s): {1 / (eta * evals.max()):.0f} ... {1 / (eta * evals.min()):.0f} steps")
print(f"max |finite difference - recursion| = {np.abs(fd - rec).max():.1e}  (response scale {np.abs(fd).max():.2f})")
print(f"converged response {rec[-1]:+.4f}, IF {IF:+.4f}; actual change of phi on deletion (eps = 1): {phi(w_del) - phi(w_star):+.4f}")
T, K = Kmax, 200
M = np.linalg.matrix_power(np.eye(d) - eta * Hbar, K)
full = M @ R[T - K] + (np.eye(d) - M) @ np.linalg.solve(Hbar, grad_Lk(w_star)) / n
print(f"window of K = {K} steps before T: |r_T - formula| = {np.linalg.norm(R[T] - full):.1e}; "
      f"with the inherited term dropped: {np.linalg.norm(R[T] - full + M @ R[T - K]):.1e}; |r_T| = {np.linalg.norm(R[T]):.2f}")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(np.arange(1, Kmax + 1), rec[1:], color=COL[0], label="unrolled recursion (Step C1)")
ax.plot(Ks, fd[Ks], "o", ms=4, mfc="none", color=COL[0], label="finite differences of retraining")
ax.plot(Ks, frozen, "--", color=COL[1], label="frozen-curvature filter")
ax.axhline(IF, color=GREY, lw=1, label="IF")
ax.set_xscale("log")
ax.set_xlabel("training steps $K$")
ax.set_ylabel(r"$d\phi/d\varepsilon$ (downweighting)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
]

# ======================================================================= stop 3
CELLS += [
    markdown(
        r"""
<a id="da2-stop-3"></a>

## 3. Bayesian influence functions

**Question.** What if training produced a distribution rather than a point? Take
the local posterior, or Gibbs measure, $p_\beta(w)\propto\exp\!\big(-\sum_j\beta_jL_j(w)-\frac\gamma2\|w-w^*\|^2\big)$:
the Gaussian factor localizes around the checkpoint $w^*$ and takes the place of a prior
(a Gaussian prior of precision $\alpha$ would just add $\alpha$ to $\gamma$). Then $\pi(\beta)$ is that local posterior, the
observable is its average $\mathbb E_{p_\beta}\phi$, and the **Bayesian influence
function** is $\mathrm{BIF}(z_i,\phi)=\partial\,\mathbb E_{p_\beta}[\phi]/\partial\beta_i$ at $\beta=\mathbf 1$.

**How.** The derivative of a Gibbs average is a covariance,
$\mathrm{BIF}(z_i,\phi)=-\mathrm{Cov}_p(L_i,\phi)$ ([Walkthrough 3A](#da2-bif-a)). No Hessian is formed
or inverted: sample $w$ from the local posterior, usually with stochastic
gradient Langevin dynamics (SGLD), and compute sample covariances for all
training examples at once.

**Result.** In a Gaussian approximation of the posterior, the leading term of
the local BIF is the damped IF, with the localization $\gamma$ as the damping
([Walkthrough 3B](#da2-bif-b)). Beyond that order the BIF sees non-quadratic
geometry, such as curved or degenerate valleys, that the IF cannot: in the
code below, a curved two-parameter valley makes the local BIF twice the damped
IF at small $\gamma$. Kreer et al. report LDS on CIFAR-10 comparable to EK-FAC.
**Caveat:** the estimate is only as good as the sampler. SGLD mixing, its
step-size bias and the choice of $\gamma$ are the practical risks. Langevin
dynamics and Gibbs laws are in [TD §2.3](../td/td2_3_sgd_sde.html#td23-3).
"""
        + KREER + MLODOZENIEC
        + "\n\n**Also:**\n" + BAKER
        + r"""
<a id="da2-bif-a"></a>

### Walkthrough 3A: the BIF is a covariance

*Substitutes for* Kreer et al. §2.2–2.3. *Shared objective:* derive
$\mathrm{BIF}=-\mathrm{Cov}$, and see that Bayesian linear regression gives the damped IF
exactly, with the damping in stated units.

**Notation.** $p_\beta(w)=e^{-\sum_j\beta_jL_j(w)}\rho(w)/Z(\beta)$ with
$Z(\beta)=\int e^{-\sum_j\beta_jL_j}\rho\,dw$; $\mathbb E$ and $\mathrm{Cov}$ are under $p=p_{\mathbf 1}$.
A localization factor $e^{-\frac\gamma2\|w-w^*\|^2}$ can be absorbed into $\rho$.

**Step A1. Differentiate a Gibbs average.** $\beta_i$ appears in the weight of
every $w$ and in the normalization.

*Exercise.* Show $\partial_{\beta_i}\log Z=-\mathbb E[L_i]$ and
$\partial_{\beta_i}\mathbb E_{p_\beta}[\phi]\big|_{\beta=\mathbf 1}=-\mathrm{Cov}_p(L_i,\phi)$.
"""
    ),
    answer(
        r"""
$\partial_{\beta_i}Z=\int(-L_i)\,e^{-\sum_j\beta_jL_j}\rho\,dw=-Z\,\mathbb E_{p_\beta}[L_i]$. Then, by the quotient rule,

$$
\partial_{\beta_i}\frac{\int\phi\,e^{-\sum_j\beta_jL_j}\rho}{Z}
=\frac{\int\phi\,(-L_i)\,e^{-\sum_j\beta_jL_j}\rho}{Z}-\frac{\partial_{\beta_i}Z}{Z}\,\mathbb E_{p_\beta}[\phi]
=-\mathbb E[\phi L_i]+\mathbb E[L_i]\,\mathbb E[\phi]=-\mathrm{Cov}(L_i,\phi).
$$

This is a fluctuation–response relation: the response of an average to the field $\beta_i$
equals minus the covariance with the conjugate energy $L_i$. The localization factor does not
involve $\beta$, so the local BIF is the same covariance under the local posterior.
"""
    ),
    markdown(
        r"""
**Step A2. Bayesian linear regression.** The one case where everything is
Gaussian. It fixes the units of the damping.

*Exercise.* Let $y_j=x_j^\top w+\text{noise}$, noise $\mathcal N(0,\sigma^2)$, prior
$\mathcal N(0,\tau^2I)$, and $L_j(w)=(y_j-x_j^\top w)^2/(2\sigma^2)$. The posterior is $\mathcal N(\mu,V)$
with $A=\sum_jx_jx_j^\top$, $V=(A/\sigma^2+I/\tau^2)^{-1}$ and $\mu=V\sum_jx_jy_j/\sigma^2$. Let
$e_j=y_j-x_j^\top\mu$.
1. Show $\mathrm{BIF}(z_j,w)=Vx_je_j/\sigma^2=(A+\tfrac{\sigma^2}{\tau^2}I)^{-1}x_je_j$ (the observable is $\phi(w)=w$, one entry at a time).
2. Write this as a damped IF in two systems of units: unscaled squared errors
   $\ell_j=\frac12(y_j-x_j^\top w)^2$, and the likelihood losses $L_j$.
"""
    ),
    answer(
        r"""
**1.** Write $w=\mu+\Delta$ with $\Delta\sim\mathcal N(0,V)$. Then
$L_j=\big(e_j^2-2e_jx_j^\top\Delta+(x_j^\top\Delta)^2\big)/(2\sigma^2)$. Constants drop out of a
covariance, and $\mathrm{Cov}\big((x_j^\top\Delta)^2,\Delta\big)$ is a third central moment of a Gaussian,
so zero. Hence
$\mathrm{Cov}(L_j,w)=-\frac{e_j}{\sigma^2}\mathrm{Cov}(x_j^\top\Delta,\Delta)=-\frac{e_j}{\sigma^2}Vx_j$, and
$\mathrm{BIF}=Vx_je_j/\sigma^2$. Multiplying $V^{-1}=(A+\frac{\sigma^2}{\tau^2}I)/\sigma^2$ out gives the second form.

**2.** In squared-error units, $\nabla\ell_j(\mu)=-x_je_j$ and the curvature is $A$, so
$\mathrm{BIF}=-(A+\lambda I)^{-1}\nabla\ell_j(\mu)$ with $\lambda=\sigma^2/\tau^2$: a damped IF at the posterior
mean, damping noise variance over prior variance. In likelihood units, $\nabla L_j(\mu)=-x_je_j/\sigma^2$
and $H=A/\sigma^2$, so $\mathrm{BIF}=-(H+\tau^{-2}I)^{-1}\nabla L_j(\mu)$: the damping is the prior precision
$\tau^{-2}$. So what counts as "the damping" depends on the loss scale: $\sigma^2/\tau^2$ for
squared errors, $1/\tau^2$ for likelihood losses.
"""
    ),
    markdown(
        r"""
**Step A3. Why it is exact here.** Three things made Step A2 exact: a Gaussian
posterior, quadratic $L_j$, and a linear $\phi$. Here we break the last one.

*Exercise.* Keep the Gaussian posterior of Step A2 but take a quadratic observable
$\phi(w)=\frac12w^\top H_\phi w$. Show that the BIF now has an extra term
$-\frac12\mathrm{tr}(H_\phi VH_jV)$ with $H_j=x_jx_j^\top/\sigma^2$, which the damped IF misses.
"""
    ),
    answer(
        r"""
With $\Delta\sim\mathcal N(0,V)$, $\phi=\phi(\mu)+\mu^\top H_\phi\Delta+\frac12\Delta^\top H_\phi\Delta$ and
$L_j=L_j(\mu)+\nabla L_j(\mu)^\top\Delta+\frac12\Delta^\top H_j\Delta$. Covariances of a linear and a quadratic
term are odd moments, so zero. The linear–linear term gives the damped IF with
$\nabla\phi(\mu)=H_\phi\mu$. The quadratic–quadratic term is, by Isserlis' theorem,
$\mathrm{Cov}(\frac12\Delta^\top H_\phi\Delta,\frac12\Delta^\top H_j\Delta)=\frac12\mathrm{tr}(H_\phi VH_jV)$. So the BIF
equals the damped IF exactly when the posterior is Gaussian (conjugacy) **and** the
covariance has no higher terms, which here needs $L_j$ quadratic and $\phi$ linear.
"""
    ),
    markdown(
        r"""
> **Scope.** $\mathrm{BIF}=-\mathrm{Cov}$ holds whenever we may differentiate under the integral. Exact agreement with the damped IF is special to Gaussian posteriors with quadratic losses and linear observables. Nothing here says that a sampler reaches the posterior.

<a id="da2-bif-b"></a>

### Walkthrough 3B: the Gaussian-surrogate expansion

*Substitutes for* Kreer et al. Appendix A. *Shared objective:* see the damped
IF as the leading term of the local BIF, and see what enters at the next order.

**Notation.** $w^*$ is a local minimum of $L=\sum_jL_j$ with $H=\nabla^2L(w^*)\succ0$.
We take the dimension fixed and every eigenvalue of $H$ of order $n$
($cnI\preceq H\preceq CnI$, as when each example contributes comparable curvature),
with derivatives of order 3 and 4 also $O(n)$. Let $\Delta=w-w^*$, and write
$g_\phi,H_\phi$ and $g_i,H_i$ for the gradients and Hessians of $\phi$ and $L_i$ at $w^*$
(all $O(1)$). $T_k[f]$ is the order-$k$ term of the Taylor series of $f$ at $w^*$.
The **Gaussian surrogate** is $q=\mathcal N(w^*,\Sigma)$ with $\Sigma=H^{-1}$, or
$\Sigma_\gamma=(H+\gamma I)^{-1}$ for the local posterior. Here the posterior is $p(w)\propto e^{-L(w)}\rho(w)$ with
a smooth prior $\rho$, and the local posterior is $p_\gamma(w)\propto e^{-L(w)-\frac\gamma2\|w-w^*\|^2}$ (the localization
replaces the prior, as in Kreer et al.), with $\mathrm{BIF}_\gamma=-\mathrm{Cov}_{p_\gamma}(L_i,\phi)$.

**Step B1. The surrogate.** Expanding $\log p=-L(w^*)-\frac12\Delta^\top H\Delta+O(n|\Delta|^3)+\log\rho$
and keeping the quadratic term gives $q$. The posterior has width $|\Delta|\sim n^{-1/2}$,
so the cubic term is $O(n^{-1/2})$ there and the prior's gradient is $O(1)$
against the $O(n)$ curvature: both are corrections. For the local posterior, $\log\rho$ is
replaced by $-\frac\gamma2|\Delta|^2$, which adds $\gamma I$ to the quadratic exactly.

**Step B2. Odd terms vanish, and order counting.** Under $q$, expand both
functions and use bilinearity of the covariance.

*Exercise.* Show $\mathrm{Cov}_q(\phi,L_i)=\sum_{k,m\ge1}\mathrm{Cov}_q(T_k[\phi],T_m[L_i])$, that terms with
$k+m$ odd vanish, and that the $(k,m)$ term is $O(n^{-(k+m)/2})$.
"""
    ),
    answer(
        r"""
Constant terms drop out of covariances, and $\mathrm{Cov}$ is bilinear. $T_k[\phi]$ is a degree-$k$
polynomial in $\Delta$, so $\mathrm{Cov}(T_k,T_m)=\mathbb E[T_kT_m]-\mathbb E[T_k]\mathbb E[T_m]$ involves moments of
$\Delta$ of total degree $k+m$. A centered Gaussian has zero odd moments. If $k+m$ is odd, then
$\mathbb E[T_kT_m]=0$, and one of $k,m$ is odd, so $\mathbb E[T_k]\mathbb E[T_m]=0$. By Isserlis' theorem a
degree-$(k+m)$ moment is a sum of products of $(k+m)/2$ entries of $\Sigma=O(n^{-1})$.
"""
    ),
    markdown(
        r"""
**Step B3. The leading term.** The only $O(n^{-1})$ term is $(1,1)$.

*Exercise.* Show $\mathrm{Cov}_q(g_\phi^\top\Delta,g_i^\top\Delta)=g_\phi^\top\Sigma g_i$. Conclude
$$
\mathrm{BIF}(z_i,\phi)\approx-g_\phi^\top H^{-1}g_i,\qquad \mathrm{BIF}_\gamma(z_i,\phi)\approx-g_\phi^\top(H+\gamma I)^{-1}g_i ,
$$
the classical and the damped IF, with $\gamma$ in the same (summed-loss) units as $\lambda$.
"""
    ),
    answer(
        r"""
$\mathrm{Cov}(g_\phi^\top\Delta,g_i^\top\Delta)=g_\phi^\top\mathbb E[\Delta\Delta^\top]g_i=g_\phi^\top\Sigma g_i$. With the minus sign of
Walkthrough 3A this is $-\nabla\phi^\top H^{-1}\nabla L_i$, the IF of DA1 §3, and with $\Sigma_\gamma$ the damped IF.
The "$\approx$" covers both the higher terms of Step B4 and the difference between $q$ and the true
posterior.
"""
    ),
    markdown(
        r"""
**Step B4. The next order.** All terms with $k+m=4$ are $O(n^{-2})$, so the next
order is not one term.

*Exercise.* Show that under $q$
$$
(2,2)=\tfrac12\mathrm{tr}(H_\phi\Sigma H_i\Sigma),\qquad
(1,3)=\tfrac12\,\nabla^3L_i[\Sigma g_\phi,\Sigma],\qquad
(3,1)=\tfrac12\,\nabla^3\phi[\Sigma g_i,\Sigma],
$$
where $\nabla^3f[u,\Sigma]=\sum_{abc}(\nabla^3f)_{abc}u_a\Sigma_{bc}$. Which of them survive for a linear
observable?
"""
    ),
    answer(
        r"""
$(2,2)$: Isserlis gives three pairings of $\mathbb E[\Delta_a\Delta_b\Delta_c\Delta_d]$. The pairing
$\Sigma_{ab}\Sigma_{cd}$ reproduces $\mathbb E[T_2[\phi]]\,\mathbb E[T_2[L_i]]$ and cancels; the other two give
$\frac14\mathrm{tr}(H_\phi\Sigma H_i\Sigma)$ each.

$(1,3)$: $T_3[L_i]=\frac16\sum(\nabla^3L_i)_{abc}\Delta_a\Delta_b\Delta_c$ and $\mathbb E[T_3]=0$. In
$\mathbb E[(g_\phi^\top\Delta)\Delta_a\Delta_b\Delta_c]$ the index of $g_\phi$ pairs with any of the three (3 ways),
each giving $(\Sigma g_\phi)_a\Sigma_{bc}$ by symmetry of $\nabla^3L_i$; so $\frac36=\frac12$. $(3,1)$ is the same with roles swapped.

For linear $\phi$, $H_\phi=0$ and $\nabla^3\phi=0$, so $(2,2)$ and $(3,1)$ vanish but $(1,3)$ does not, unless
$L_i$ is quadratic. A linear observable does not remove the corrections.
"""
    ),
    markdown(
        r"""
**The true posterior at this order.** The surrogate drops the cubic part of $L$
and the prior. These skew and shift the true posterior, and their effect on
the covariance is also $O(n^{-2})$. Example: one parameter,
$L=n\big(\frac12w^2+aw^3+bw^4\big)$, $\phi=w$, $L_i=\frac12(w-c)^2$. Every next-order
surrogate term vanishes ($\phi$ is linear, $L_i$ quadratic), yet a cumulant
expansion of the true posterior gives
$\mathrm{Cov}(\phi,L_i)=-\frac cn-\frac{6a+c(36a^2-12b)}{n^2}+O(n^{-3})$: the skewness ($a$) and
the quartic term ($b$) of $L$ enter at the same order as the surrogate terms. So
the next-order BIF needs the true posterior's corrections, not only the surrogate terms.

**When the surrogate fails.** If $H$ is singular the unlocalized surrogate does
not exist, but the BIF is still defined. The localized one,
$\mathcal N\big(w^*,(H+\gamma I)^{-1}\big)$, exists but can be badly off at weak localization:
35 times too large at $\gamma=1$ in Code 3.

> **Scope.** We expanded the covariance under a Gaussian surrogate at a nondegenerate minimum with all curvatures of order $n$, to second order. Not derived: the true posterior's corrections, which enter at the same order as the $(1,3)$, $(2,2)$ and $(3,1)$ terms, and anything for singular $H$.

<a id="da2-bif-c"></a>

### Walkthrough 3C (advanced): influence functions as optimal transport

*Substitutes for* Mlodozeniec et al. §4.2 (Theorem 3), nondegenerate case.
*Shared objective:* show that shifting each sample of the low-temperature
Gibbs distribution by the IF is the KL-optimal first-order way to reach the
downweighted distribution, and see why the partition function matters. Long;
it uses Laplace's method.

**Notation.** $\kappa>0$ is an inverse temperature ($\beta$ stays the data weights).
$$
p_0(w)=\frac{e^{-\kappa L(w)}}{Z_0},\qquad p_\varepsilon(w)=\frac{e^{-\kappa(L(w)-\varepsilon L_k(w))}}{Z_\varepsilon}:
$$
$z_k$ downweighted by $\varepsilon$. $w^*$ is the unique global minimum of $L$, with
$H=\nabla^2L(w^*)\succ0$ and $g_k=\nabla L_k(w^*)$. A smooth, bounded vector field $r$ defines the
transport $T_\varepsilon(w)=w+\varepsilon r(w)$, assumed to be a diffeomorphism for small
$\varepsilon$ (bounded $\nabla r$ suffices); $q_\varepsilon$ is the distribution of $T_\varepsilon(w)$ for
$w\sim p_0$. The cost is $\mathcal F(r,\varepsilon)=\lim_{\kappa\to\infty}\frac1\kappa\mathrm{KL}(q_\varepsilon\,\|\,p_\varepsilon)$.

**Step C1. KL in terms of energies.** Change variables back to $w\sim p_0$.

*Exercise.* Show
$$
\frac1\kappa\mathrm{KL}(q_\varepsilon\|p_\varepsilon)=\mathbb E_{p_0}\big[L(w+\varepsilon r)-\varepsilon L_k(w+\varepsilon r)-L(w)\big]+\frac1\kappa\log\frac{Z_\varepsilon}{Z_0}-\frac1\kappa\mathbb E_{p_0}\log\det(I+\varepsilon\nabla r).
$$
"""
    ),
    answer(
        r"""
For a diffeomorphism $T$, $q(T(w))=p_0(w)/|\det\nabla T(w)|$. So
$\mathrm{KL}(q\|p_\varepsilon)=\mathbb E_{\tilde w\sim q}[\log q(\tilde w)-\log p_\varepsilon(\tilde w)]
=\mathbb E_{w\sim p_0}\big[\log p_0(w)-\log\det(I+\varepsilon\nabla r)-\log p_\varepsilon(w+\varepsilon r)\big]$
($\det>0$ for small $\varepsilon$). Insert $\log p_0=-\kappa L-\log Z_0$ and
$\log p_\varepsilon=-\kappa(L-\varepsilon L_k)-\log Z_\varepsilon$, and divide by $\kappa$.
"""
    ),
    markdown(
        r"""
**Step C2. Low temperature.** As $\kappa\to\infty$, $p_0$ concentrates at $w^*$, the
log-determinant is $O(1)$, and Laplace's method gives
$\frac1\kappa\log Z_\varepsilon\to-\inf_w\big(L-\varepsilon L_k\big)$.

*Exercise.* Show
$\inf_w(L-\varepsilon L_k)=L(w^*)-\varepsilon L_k(w^*)-\frac{\varepsilon^2}2g_k^\top H^{-1}g_k+O(\varepsilon^3)$, so the
partition-function term tends to $\varepsilon L_k(w^*)+\frac{\varepsilon^2}2g_k^\top H^{-1}g_k+O(\varepsilon^3)$, which is
not zero.
"""
    ),
    answer(
        r"""
Near $w^*$, $L(w^*+\delta)-\varepsilon L_k(w^*+\delta)=L(w^*)-\varepsilon L_k(w^*)+\frac12\delta^\top H\delta-\varepsilon g_k^\top\delta+O(|\delta|^3+\varepsilon|\delta|^2)$.
The minimizer is $\delta^*=\varepsilon H^{-1}g_k+O(\varepsilon^2)$, and the minimum value is
$L(w^*)-\varepsilon L_k(w^*)-\frac{\varepsilon^2}2g_k^\top H^{-1}g_k+O(\varepsilon^3)$. For small $\varepsilon$ the global infimum is
this local one, because $w^*$ is the unique global minimum. Then
$\frac1\kappa\log(Z_\varepsilon/Z_0)\to\inf L-\inf(L-\varepsilon L_k)=\varepsilon L_k(w^*)+\frac{\varepsilon^2}2g_k^\top H^{-1}g_k+O(\varepsilon^3)$.
"""
    ),
    markdown(
        r"""
**Step C3. Complete the square.**

*Exercise.* Combine Steps C1–C2 to show
$$
\mathcal F(r,\varepsilon)=\frac{\varepsilon^2}2\big(r(w^*)-H^{-1}g_k\big)^\top H\big(r(w^*)-H^{-1}g_k\big)+O(\varepsilon^3).
$$
What would go wrong without the partition-function term?
"""
    ),
    answer(
        r"""
The expectation concentrates at $w^*$, where $\nabla L=0$. With $r=r(w^*)$:
$L(w^*+\varepsilon r)-L(w^*)=\frac{\varepsilon^2}2r^\top Hr+O(\varepsilon^3)$ and
$\varepsilon L_k(w^*+\varepsilon r)=\varepsilon L_k(w^*)+\varepsilon^2g_k^\top r+O(\varepsilon^3)$. Adding the partition term, the
$\varepsilon L_k(w^*)$ terms cancel and

$$
\mathcal F=\frac{\varepsilon^2}2\big(r^\top Hr-2g_k^\top r+g_k^\top H^{-1}g_k\big)+O(\varepsilon^3)
=\frac{\varepsilon^2}2(r-H^{-1}g_k)^\top H(r-H^{-1}g_k)+O(\varepsilon^3).
$$

Without the partition term, $\mathcal F$ would contain $-\varepsilon L_k(w^*)$ at first order and, even
ignoring that, its minimum over $r$ would be $-\frac{\varepsilon^2}2g_k^\top H^{-1}g_k<0$: impossible for a KL.
The partition function supplies exactly the constant that completes the square.
"""
    ),
    markdown(
        r"""
**Step C4. What is determined.** The optimum is $r(w^*)=H^{-1}g_k$: the first-order
displacement of the minimizer under downweighting, which is minus the upweighting
parameter IF. The cost depends on $r$ only through its value at $w^*$, so any
smooth field with that value is optimal at this order. "The IF is the optimal
transport" means the optimal *value at the minimum*, not a unique vector field.

**Step C5. A finite-temperature check.** Bayesian linear regression at $\kappa=1$:
$L_j=(y_j-x_j^\top w)^2/(2\sigma^2)$, prior $\mathcal N(0,\tau^2I)$, posterior exactly $\mathcal N(\mu,V)$ with
$V^{-1}=\sum_jx_jx_j^\top/\sigma^2+I/\tau^2$, residuals $e_j=y_j-x_j^\top\mu$, and (Walkthrough 3A)
$\mathrm{BIF}(z_k,w)=Vx_ke_k/\sigma^2$.

*Exercise.* Downweighting $z_k$ by $\varepsilon$ gives $\mathcal N(\mu_\varepsilon,V_\varepsilon)$ with
$V_\varepsilon^{-1}=V^{-1}-\varepsilon x_kx_k^\top/\sigma^2$. A constant shift sends $\mathcal N(\mu,V)$ to $\mathcal N(\mu+\varepsilon r,V)$.
1. Show $\mu_\varepsilon=\mu-\varepsilon\,\mathrm{BIF}(z_k,w)+O(\varepsilon^2)$.
2. With $M=x_kx_k^\top V/\sigma^2$, show
   $\mathrm{KL}\big(\mathcal N(\mu+\varepsilon r,V)\,\|\,\mathcal N(\mu_\varepsilon,V_\varepsilon)\big)=\frac{\varepsilon^2}2(r+\mathrm{BIF})^\top V^{-1}(r+\mathrm{BIF})+\frac{\varepsilon^2}4\mathrm{tr}(M^2)+O(\varepsilon^3)$.
3. Check consistency with Step C3: $V^{-1}$ is the Hessian of the energy, and $V^{-1}(-\mathrm{BIF})=\nabla L_k(\mu)$.
"""
    ),
    answer(
        r"""
**1.** By definition $\mathrm{BIF}=\partial\mu/\partial\beta_k$ (Walkthrough 3A), and downweighting is $\beta_k=1-\varepsilon$.

**2.** For Gaussians, $\mathrm{KL}(\mathcal N(m_0,V)\|\mathcal N(m_1,V_\varepsilon))=\frac12\big[\mathrm{tr}(V_\varepsilon^{-1}V)-d+(m_1-m_0)^\top V_\varepsilon^{-1}(m_1-m_0)+\log\frac{\det V_\varepsilon}{\det V}\big]$.
Here $V_\varepsilon^{-1}V=I-\varepsilon M$, so $\mathrm{tr}(V_\varepsilon^{-1}V)-d=-\varepsilon\,\mathrm{tr}M$ and
$\log\frac{\det V_\varepsilon}{\det V}=-\log\det(I-\varepsilon M)=\varepsilon\,\mathrm{tr}M+\frac{\varepsilon^2}2\mathrm{tr}M^2+O(\varepsilon^3)$.
The covariance part is $\frac{\varepsilon^2}4\mathrm{tr}M^2$. The mean part is
$\frac12\varepsilon^2(r+\mathrm{BIF})^\top V^{-1}(r+\mathrm{BIF})+O(\varepsilon^3)$, using $m_1-m_0=-\varepsilon(\mathrm{BIF}+r)+O(\varepsilon^2)$
and $V_\varepsilon^{-1}=V^{-1}+O(\varepsilon)$. So the best constant shift is $r=-\mathrm{BIF}$, as in Step C4, but the
minimal cost is $\frac{\varepsilon^2}4\mathrm{tr}M^2\neq0$: a shift cannot reproduce the change of covariance.
At low temperature this term is $O(1)$ in KL against the $O(\kappa)$ energy term, which is
why it disappears from $\mathcal F$.

**3.** The energy is $\sum_jL_j+\|w\|^2/(2\tau^2)$, with Hessian $A/\sigma^2+I/\tau^2=V^{-1}$. And
$V^{-1}(-\mathrm{BIF})=V^{-1}(-Vx_ke_k/\sigma^2)=-x_ke_k/\sigma^2=\nabla L_k(\mu)$, matching $r=H^{-1}g_k$.
"""
    ),
    markdown(
        r"""
**Step C6. Degenerate minima.** If $H$ is singular, Step C2 needs care. If
$g_k$ has a component along a flat direction, $L-\varepsilon L_k$ can decrease faster than
$\varepsilon^2$ along it: for $L=w^4$ and $L_k=w$, $\inf(L-\varepsilon L_k)=-3(\varepsilon/4)^{4/3}$. The completed
square, with $H^+$ in place of $H^{-1}$, needs $g_k\in\mathrm{Range}(H)$.

> **Scope.** Proved: a nondegenerate unique minimum and the low-temperature normalized KL. At finite temperature (Step C5) the covariance change costs $O(\varepsilon^2)$ that no shift removes. Theorem 3 of Mlodozeniec et al. treats sets of minima; degenerate directions need at least $g_k\in\mathrm{Range}(H)$ (Step C6), and there the optimal field is not unique.

<a id="da2-bif-code"></a>

### Code 3: SGLD-BIF against quadrature

*Part 1: calibrate the covariance estimator.* In Bayesian linear regression
(Walkthrough 3A, Step A2, with $n=30$, $d=3$) the BIF is known exactly. We draw
**exact** posterior samples, with no SGLD, so any error is Monte Carlo
error, and check that the error bars computed from 20 batches are honest:
about 95% of the $|z|$-scores should be below 2, and the error should fall like
$N^{-1/2}$.
"""
    ),
    code(
        r"""
t0 = time.time()
rng = np.random.default_rng(1)
n_b, d_b, sig, tau = 30, 3, 0.5, 1.3
Xb = rng.standard_normal((n_b, d_b))
yb = Xb @ rng.standard_normal(d_b) + sig * rng.standard_normal(n_b)
V = np.linalg.inv(Xb.T @ Xb / sig**2 + np.eye(d_b) / tau**2)
mu_b = V @ Xb.T @ yb / sig**2
bif_exact = (V @ (Xb * (yb - Xb @ mu_b)[:, None]).T / sig**2).T     # (n, d): V x_j e_j / sigma^2
def bif_estimate(N, G=20):
    # -Cov(L_j, w) for all j from N exact posterior draws, with a standard error from G batches
    Wb = rng.multivariate_normal(mu_b, V, size=N)
    Lb = (yb[None, :] - Wb @ Xb.T) ** 2 / (2 * sig**2)                # (N, n) per-example losses
    batches = []
    for B in np.array_split(np.arange(N), G):
        Lc, Wc = Lb[B] - Lb[B].mean(0), Wb[B] - Wb[B].mean(0)
        batches.append(-(Lc.T @ Wc) / (len(B) - 1))
    batches = np.array(batches)
    return batches.mean(0), batches.std(0, ddof=1) / np.sqrt(G)


for N in [10**3, 10**4, 10**5]:
    est, se = bif_estimate(N)
    print(f"N = {N:>6}: rms error {np.sqrt(((est - bif_exact) ** 2).mean()):.1e}, median standard error {np.median(se):.1e}")
zs = np.concatenate([((lambda e, s: (e - bif_exact) / s)(*bif_estimate(10**4))).ravel() for _ in range(30)])
print(f"30 replicates at N = 10^4: fraction of |z| < 2 = {(np.abs(zs) < 2).mean():.3f} "
      f"(t with 19 d.o.f.: 0.940), rms z = {np.sqrt((zs**2).mean()):.2f}")
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    markdown(
        r"""
*Part 2: a curved valley.* Five data points $z_j$ and the model $u=ab$, with
$L_j(a,b)=\frac12(z_j-ab)^2$: the two-parameter model of the SLT day, whose minima
form the hyperbola $ab=\bar z$. Localize at the balanced minimum
$w^*=(\sqrt{\bar z},\sqrt{\bar z})$ with strength $\gamma$. The observable is the loss on
a query $z_q=\bar z+0.6$; we report the BIF of the training point farthest from
$\bar z$. For each $\gamma$ we compare:
- **quadrature**: the exact local BIF, by summing over a fine grid in $(a,b)$;
- **SGLD** (here plain, full-batch Langevin) at step sizes $h$ and $h/2$, with 256 chains started at $w^*$, run for 64 relaxation times along the valley (a quarter discarded as burn-in), and the Monte Carlo error from 16 groups of chains;
- the **damped IF** $-\nabla\phi^\top(H+\gamma I)^{-1}\nabla L_i$ at $w^*$;
- the **Gaussian surrogate**: the same covariance under $\mathcal N(w^*,(H+\gamma I)^{-1})$, by quadrature.

Look for where the local BIF departs from
the damped IF, whether SGLD tracks quadrature within its error bars, and
whether halving $h$ moves the estimate.
"""
    ),
    code(
        r"""
t0 = time.time()
rng = np.random.default_rng(0)
z = 1.5 + 0.5 * rng.standard_normal(5)
n_z, zbar = len(z), z.mean()
w_star = np.full(2, np.sqrt(zbar))
zq = zbar + 0.6
i = int(np.argmax(np.abs(z - zbar)))
v = w_star[::-1]                                    # grad of ab at w* is (b, a)
H = n_z * np.outer(v, v)                            # Hessian of sum_j L_j at the minimum (rank one)
L_i = lambda a, b: 0.5 * (z[i] - a * b) ** 2
phi = lambda a, b: 0.5 * (zq - a * b) ** 2


def cov_on_grid(gamma, density):
    half = 7 / np.sqrt(gamma) + 1.0
    ax_ = np.linspace(-half, half, 801)
    A, B = np.meshgrid(w_star[0] + ax_, w_star[1] + ax_, indexing="ij")
    p = density(A, B)
    p = p / p.sum()
    Li, ph = L_i(A, B), phi(A, B)
    return -((p * Li * ph).sum() - (p * Li).sum() * (p * ph).sum())


def local_posterior(gamma):
    def dens(A, B):
        U = 0.5 * ((z[:, None, None] - A * B) ** 2).sum(0) + 0.5 * gamma * ((A - w_star[0]) ** 2 + (B - w_star[1]) ** 2)
        return np.exp(-(U - U.min()))
    return dens


def surrogate(gamma):
    Prec = H + gamma * np.eye(2)
    return lambda A, B: np.exp(-0.5 * (Prec[0, 0] * (A - w_star[0]) ** 2 + 2 * Prec[0, 1] * (A - w_star[0]) * (B - w_star[1])
                                       + Prec[1, 1] * (B - w_star[1]) ** 2))


def sgld(gamma, h, chains=256, seed=1):
    # Langevin on U = sum_j L_j + gamma/2 |w - w*|^2: w <- w - (h/2) grad U + sqrt(h) xi
    r = np.random.default_rng(seed)
    w = np.tile(w_star, (chains, 1))
    steps = int(128 / (h * gamma))     # the slowest relaxation (along the valley) takes ~2/(h gamma) steps
    draws = []
    for t in range(steps):
        a, b = w[:, 0], w[:, 1]
        res = ((a * b)[:, None] - z[None, :]).sum(1)
        grad_U = np.stack([res * b, res * a], 1) + gamma * (w - w_star)
        w = w - 0.5 * h * grad_U + np.sqrt(h) * r.standard_normal(w.shape)
        if t >= steps // 4 and t % 4 == 0:
            draws.append(w.copy())
    W = np.array(draws)                                      # (draws, chains, 2)
    groups = []
    for cs in np.array_split(np.arange(chains), 16):
        a, b = W[:, cs, 0].ravel(), W[:, cs, 1].ravel()
        groups.append(-np.cov(L_i(a, b), phi(a, b))[0, 1])
    return np.mean(groups), np.std(groups, ddof=1) / 4, steps


gammas = [1, 3, 10, 30, 100, 300]
rows = []
print(f"{'gamma':>6} {'quadrature':>11} {'SGLD, h':>19} {'SGLD, h/2':>19} {'damped IF':>10} {'Gaussian surr.':>15}")
for gamma in gammas:
    quad = cov_on_grid(gamma, local_posterior(gamma))
    surr = cov_on_grid(gamma, surrogate(gamma))
    damped = -((-(zq - zbar) * v) @ np.linalg.solve(H + gamma * np.eye(2), -(z[i] - zbar) * v))
    h = 0.1 / (n_z * v @ v + gamma)
    s1, e1, _ = sgld(gamma, h)
    s2, e2, _ = sgld(gamma, h / 2, seed=2)
    rows.append((quad, s1, e1, s2, e2, damped, surr))
    print(f"{gamma:>6} {quad:>11.5f} {s1:>9.5f} ± {e1:.5f} {s2:>9.5f} ± {e2:.5f} {damped:>10.5f} {surr:>15.5f}")
rows = np.array(rows)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(gammas, -rows[:, 0], color=COL[0], label="local BIF, quadrature")
ax.errorbar(np.array(gammas) * 0.96, -rows[:, 1], yerr=2 * rows[:, 2], fmt="o", ms=4, color=COL[0], mfc="none", label=r"SGLD, $h$ ($\pm2$ s.e.)")
ax.errorbar(np.array(gammas) * 1.04, -rows[:, 3], yerr=2 * rows[:, 4], fmt="s", ms=4, color=COL[2], mfc="none", label=r"SGLD, $h/2$")
ax.plot(gammas, -rows[:, 5], "--", color=COL[1], label=r"damped IF, $\lambda=\gamma$")
ax.plot(gammas, -rows[:, 6], ":", color=GREY, label="Gaussian surrogate")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-3, 4)
ax.set_xlabel(r"localization $\gamma$")
ax.set_ylabel(r"$-$BIF (helpfulness to the query)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
    markdown(
        r"""
**What the run shows.** At $\gamma\le3$ the local BIF is about twice the damped IF:
the curved valley matters, and the IF's quadratic picture misses it. The
Gaussian surrogate is much worse there (35 times too large at $\gamma=1$),
because it extends the posterior straight along the flat direction instead of
along the curved valley. By $\gamma=30$ the three agree to within about 10–15%,
and by $\gamma=100$ to a few percent. At step $h$, SGLD is 0.5–3% too large in
magnitude at every $\gamma$ (1–2 standard errors each); halving $h$ brings it
closer in five of the six rows: a small step-size bias. Chains started at $w^*$ need a burn-in long compared with
the slowest relaxation, $\sim2/(h\gamma)$ steps along the valley; too short a
burn-in leaves them under-dispersed and biases the covariance toward zero.
"""
    ),
]

# ======================================================================= stop 4
CELLS += [
    markdown(
        r"""
<a id="da2-stop-4"></a>

## 4. Influence changes over training

**Question.** Ask the attribution question at every checkpoint: how does
upweighting $z_p$ change a query loss $\ell_q(w_t)$, as a function of $t$? Here
$\pi(\beta)=w_t(\beta)$ for each $t$, as in unrolling, and we watch the answer
evolve.

**Result.** Attribution is a time series, not a number. Networks learn coarse
structure before fine structure ([TD NB1](../td/td1_linear_networks.html#td1-s1)). In a deep linear network learning a
small hierarchy, an example's influence on a related query is helpful while
their shared coarse structure is learned, harmful on the plateau before the
fine structure is learned, and zero at the end ([Walkthrough 4](#da2-dln),
[Code 4](#da2-dln-code)). Lee et
al. report the same kind of sign flips in a toy hierarchy and in a small
language model, using the BIF at each checkpoint. **Caveat:** an endpoint
influence function, however exact, sees none of this. In the walkthrough's
dog/cat example (realizable data) it is exactly zero.
"""
        + LEE
        + r"""
<a id="da2-dln"></a>

### Walkthrough 4: influence in a deep linear network

*Substitutes for* the derivation in Lee et al. Appendix C.6. *Shared objective:* derive how
upweighting one example changes a two-layer linear network along its
trajectory, separate a "speed" channel from a "rotation" channel, and explain a
sign flip. *Needs* the mode solution of [TD NB1 §1](../td/td1_linear_networks.html#td1-s1),
restated here.

**Notation.** A two-layer linear network $f(x)=W_2W_1x$ with product $W=W_2W_1$ (all
matrices $d\times d$). Data $(x_j,y_j)$, $j=1,\dots,n$, with weights $\beta_j$; mean loss
$L(W;\beta)=\frac1n\sum_j\beta_j\frac12\|y_j-Wx_j\|^2$. With
$\Sigma_{xx}(\beta)=\frac1n\sum_j\beta_jx_jx_j^\top$ and $\Sigma_{yx}(\beta)=\frac1n\sum_j\beta_jy_jx_j^\top$,
gradient flow is
$$
\tau\dot W_1=W_2^\top E,\qquad \tau\dot W_2=EW_1^\top,\qquad E=\Sigma_{yx}-W\Sigma_{xx},
$$
with $\tau=1/\eta$, so $t/\tau=\eta\times(\text{number of steps})$. At $\beta=\mathbf 1$ the inputs are whitened,
$\Sigma_{xx}=I$, and $\Sigma_{yx}=USV^\top$ with singular values $s_1>s_2>\dots>0$. From the
aligned, balanced, small initialization $W_1(0)=\sqrt{a_0}\,V^\top$, $W_2(0)=\sqrt{a_0}\,U$,
$$
W(t)=U\,\mathrm{diag}\big(a_k(t)\big)V^\top,\qquad \tau\dot a_k=2a_k(s_k-a_k),\qquad
a_k(t)=\frac{s_ke^{2s_kt/\tau}}{e^{2s_kt/\tau}-1+s_k/a_0}.
$$
Mode $k$ rises around $t_k^*\approx\frac{\tau}{2s_k}\ln(s_k/a_0)$, strongest first. We work in
the mode basis ($x\to V^\top x$, $y\to U^\top y$, $W\to U^\top WV$), which amounts to $U=V=I$.

We upweight example $p$, $\beta_p=1+\varepsilon$, keeping the initialization fixed. Write
$\delta W(t)=\partial W(t)/\partial\beta_p$ at $\beta=\mathbf 1$. The influence on the query loss
$\ell_q=\frac12\|y_q-Wx_q\|^2$ is
$$
\mathcal I_q(t)=-r_q(t)^\top\,\delta W(t)\,x_q,\qquad r_q(t)=y_q-W(t)x_q .
$$

**Step 1. An example acts through its residual.** Reweighting changes both
covariances. Together they enter as a forcing proportional to the example's
*current* residual, so an example the network already fits exerts no force.

*Exercise.* Show that at $\beta=\mathbf 1$, $\partial E/\partial\beta_p=F(t)-\delta W$ with
$F(t)=\frac1nr_p(t)x_p^\top$ and $r_p=y_p-W(t)x_p$. Write the linear equations for
$\delta W_1,\delta W_2$ and their initial condition.
"""
    ),
    answer(
        r"""
$\partial\Sigma_{yx}/\partial\beta_p=\frac1ny_px_p^\top$ and $\partial\Sigma_{xx}/\partial\beta_p=\frac1nx_px_p^\top$. So
$\partial E/\partial\beta_p=\frac1ny_px_p^\top-\delta W\,\Sigma_{xx}-W\frac1nx_px_p^\top=\frac1n(y_p-Wx_p)x_p^\top-\delta W$,
using $\Sigma_{xx}=I$. Differentiating the flow,

$$
\tau\,\delta\dot W_1=\delta W_2^\top E+W_2^\top\delta E,\qquad
\tau\,\delta\dot W_2=\delta E\,W_1^\top+E\,\delta W_1^\top,\qquad
\delta W=\delta W_2W_1+W_2\delta W_1,
$$

with $\delta E=F-\delta W$ and $\delta W_1(0)=\delta W_2(0)=0$, since the initialization is fixed. These
are exact at linear order in $\varepsilon$. Perturbing only $\Sigma_{yx}$ would give the forcing
$\frac1ny_px_p^\top$ instead, which keeps pushing even when $p$ is fit perfectly.
"""
    ),
    markdown(
        r"""
**Step 2. Two channels.** Along the aligned path, $W_1=W_2=\mathrm{diag}(\sqrt{a_k})$ and
$E=\mathrm{diag}(s_k-a_k)$ are diagonal, so the linear equations split. The diagonal
entries of $\delta W$ (mode strengths: the *speed* channel) obey closed scalar
equations. Each off-diagonal pair $(j,k),(k,j)$ (mode mixing: the *rotation*
channel) couples only to itself.

*Exercise.* Write $P=\delta W_1$, $Q=\delta W_2$, $d_k=\sqrt{a_k}$, $e_k=s_k-a_k$. Show that
$\delta a_k:=\delta W_{kk}$ satisfies
$$
\tau\,\delta\dot a_k=2(s_k-2a_k)\,\delta a_k+2a_k\,F_{kk}(t),
$$
and that the entries $P_{jk},Q_{jk},P_{kj},Q_{kj}$ for $j\ne k$ form a closed system.
"""
    ),
    answer(
        r"""
With $D=\mathrm{diag}(d_k)$ and $E=\mathrm{diag}(e_k)$: $(Q^\top E)_{jk}=Q_{kj}e_k$, $(D\,\delta E)_{jk}=d_j\delta E_{jk}$,
$(\delta E\,D)_{jk}=\delta E_{jk}d_k$ and $(EP^\top)_{jk}=e_jP_{kj}$. So

$$
\tau\dot P_{jk}=e_kQ_{kj}+d_j\,\delta E_{jk},\qquad \tau\dot Q_{jk}=d_k\,\delta E_{jk}+e_jP_{kj},\qquad
\delta W_{jk}=d_jP_{jk}+d_kQ_{jk}.
$$

For $j\ne k$ these involve only the four entries listed. For $j=k$, let $\sigma=P_{kk}+Q_{kk}$, so
$\delta a_k=d_k\sigma$ and $\tau\dot\sigma=e_k\sigma+2d_k(F_{kk}-d_k\sigma)$; also $P_{kk}-Q_{kk}$ obeys
$\tau\frac{d}{dt}(P_{kk}-Q_{kk})=-e_k(P_{kk}-Q_{kk})$ and stays zero. A balanced mode has
$\tau\dot d_k=d_ke_k$. Then
$\tau\,\delta\dot a_k=\tau\dot d_k\sigma+d_k\tau\dot\sigma=2e_k\delta a_k-2a_k\delta a_k+2a_kF_{kk}=2(s_k-2a_k)\delta a_k+2a_kF_{kk}$.
"""
    ),
    markdown(
        r"""
**Step 3. The speed channel, solved.** Shifting time along the trajectory solves
the unforced equation: $\dot a_k$ satisfies it. So the forced equation is solved by
variation of parameters.

*Exercise.* Show
$$
\delta a_k(t)=\dot a_k(t)\int_0^t\frac{F_{kk}(t')}{s_k-a_k(t')}\,dt' .
$$
1. *A cross-covariance intervention:* replace the forcing $F$ of Step 1 by a constant
   change $\delta\Sigma_{yx}$ with diagonal entry $\delta s$, keeping $\Sigma_{xx}=I$, so $F_{kk}=\delta s$. Show $\delta a_k=(\partial a_k/\partial s_k)\,\delta s$, and
   that $\partial a_k/\partial s_k\to1$ after mode $k$ saturates.
2. *Realizable data:* $y_j=Sx_j$ for every $j$. Show
   $\delta a_k(t)=\frac{(x_p)_k^2}n\,t\,\dot a_k(t)$: upweighting $p$ runs mode $k$'s clock faster by the
   factor $1+\varepsilon(x_p)_k^2/n$.
"""
    ),
    answer(
        r"""
Differentiating $\tau\dot a=2a(s-a)$ gives $\tau\ddot a=2(s-2a)\dot a$. With $\delta a=\dot a\,I(t)$ and
$I(t)=\int_0^tF/(s-a)$:
$\tau\,\delta\dot a=\tau\ddot aI+\tau\dot a\,F/(s-a)=2(s-2a)\delta a+2a(s-a)F/(s-a)$, which is the
equation, and $\delta a(0)=0$.

**1.** Differentiate $\tau\dot a=2a(s-a)$ with respect to $s$:
$\tau\frac{d}{dt}\frac{\partial a}{\partial s}=2(s-2a)\frac{\partial a}{\partial s}+2a$, with $\partial a/\partial s=0$ at $t=0$ ($a_0$ does not
depend on $s$). That is the equation with $F=1$, so $\delta a=(\partial a/\partial s)\,\delta s$. After
saturation $a_k\to s_k$, so $\partial a_k/\partial s_k\to1$: the change in the learned strength persists.

**2.** $(r_p)_k=(s_k-a_k)(x_p)_k$, so $F_{kk}/(s_k-a_k)=(x_p)_k^2/n$ is constant and
$\delta a_k=\dot a_k\,t\,(x_p)_k^2/n$. That is the first-order term of $a_k\big((1+\varepsilon c)t\big)$ with
$c=(x_p)_k^2/n$. With realizable data the target is unchanged; only the speed is.
"""
    ),
    markdown(
        r"""
**Step 4. When is influence localized in time?** Through the speed channel the
query sees $-(r_q)_k(x_q)_k\,\delta a_k$, the product of the mode-$k$ query residual and
$\delta a_k$.

*Exercise.*
1. For realizable data and a realizable query ($y_q=Sx_q$), show that the speed
   contribution is localized around $t_k^*$ and is $\le0$: helpful.
2. For the cross-covariance intervention, $\delta a_k\to\delta s\ne0$ after saturation. What then
   makes the contribution to $\mathcal I_q$ localized?
"""
    ),
    answer(
        r"""
**1.** $\dot a_k=2a_k(s_k-a_k)/\tau$ is exponentially small before the rise ($a_k\approx a_0e^{2s_kt/\tau}$)
and after it ($s_k-a_k$ decays exponentially), with its peak at $a_k=s_k/2$, near $t_k^*$; the
factor $t$ does not change that. Also $(r_q)_k(x_q)_k=(s_k-a_k)(x_q)_k^2\ge0$ and $\delta a_k\ge0$, so
the contribution is $\le0$. Speeding up the learning of true structure helps every query.

**2.** The factor $(r_q)_k=(y_q)_k-a_k(x_q)_k$, which tends to zero for a query the network
ends up fitting. What is localized is the product of the mode-$k$ residual and the
sensitivity, not the sensitivity $\partial a_k/\partial s_k$, which tends to 1.
"""
    ),
    markdown(
        r"""
**Step 5. The derivative of a truncated SVD.** Between rises the network sits
near a *partial reconstruction*: modes $1,\dots,r$ learned ($a_k\approx s_k$), the rest
near zero. Such a point is a critical point of the loss, and near it the
reweighting acts like a change $\delta\Sigma_{yx}=F$ of the cross-covariance, with
$F=\frac1nr_px_p^\top$ from Step 1. If the network follows the moving critical
point, its rotation channel is the derivative of the truncated singular value
decomposition. This step computes that derivative.

*Exercise.*
1. *The SVD derivative.* Let $\Sigma=USV^\top$ have distinct singular values, perturb it by
   $\delta\Sigma$, and let $B=U^\top\delta\Sigma V$, $A=U^\top\delta U$, $C=V^\top\delta V$ ($A$ and $C$ are
   antisymmetric). Show $B=AS+\delta S-SC$ with $\delta S=\mathrm{diag}(B)$, and for $j\ne k$
   $$
   A_{jk}=\frac{s_kB_{jk}+s_jB_{kj}}{s_k^2-s_j^2},\qquad C_{jk}=\frac{s_jB_{jk}+s_kB_{kj}}{s_k^2-s_j^2}.
   $$
   Check: $S=\mathrm{diag}(3,1)$ with $B_{12}=1$ and all other entries 0 gives $A_{12}=-1/8$.
2. *Truncation.* For $W_r=\sum_{k\le r}s_ku_kv_k^\top$, show that for $k\le r<j$ the derivative has
   $\delta W_{jk}=s_kA_{jk}$ and $\delta W_{kj}=s_kC_{jk}$, while $\delta W_{jk}=B_{jk}$ when both $j,k\le r$.
"""
    ),
    answer(
        r"""
**1.** Differentiate $\Sigma=USV^\top$: $\delta\Sigma=\delta U\,SV^\top+U\,\delta S\,V^\top+US\,\delta V^\top$. Multiply by
$U^\top$ on the left and $V$ on the right: $B=AS+\delta S+S\,\delta V^\top V=AS+\delta S-SC$, since
$\delta V^\top V=C^\top=-C$. Diagonal entries: $A_{kk}=C_{kk}=0$, so $\delta s_k=B_{kk}$. Off the diagonal,
$B_{jk}=A_{jk}s_k-s_jC_{jk}$ and $B_{kj}=A_{kj}s_j-s_kC_{kj}=-A_{jk}s_j+s_kC_{jk}$. Multiply the first by
$s_k$, the second by $s_j$, and add: $s_kB_{jk}+s_jB_{kj}=(s_k^2-s_j^2)A_{jk}$. Multiply the first by
$s_j$, the second by $s_k$, and add: $s_jB_{jk}+s_kB_{kj}=(s_k^2-s_j^2)C_{jk}$. Check:
$A_{12}=(s_2B_{12}+s_1B_{21})/(s_2^2-s_1^2)=1/(1-9)=-1/8$.

**2.** $\delta W_r=\sum_{l\le r}\big(\delta s_l\,u_lv_l^\top+s_l\,\delta u_lv_l^\top+s_lu_l\,\delta v_l^\top\big)$. For $j>r$, $u_j^\top u_l=0$
and $v_l^\top v_j=0$, so $u_j^\top\delta W_rv_k=s_k\,u_j^\top\delta u_k=s_kA_{jk}$ and
$u_k^\top\delta W_rv_j=s_k\,\delta v_k^\top v_j=s_kC_{jk}$. When $j,k\le r$ all terms survive and give the full
$B=AS+\delta S-SC$ entry, $B_{jk}$.
"""
    ),
    markdown(
        r"""
**Step 6. The rotation channel on a plateau.** Does the network actually follow
the critical point? On a plateau the off-diagonal equations of Step 2 have
nearly constant coefficients and relax quickly, so the rotation channel settles
at their fixed point. We check that this fixed point is Step 5's truncated SVD
derivative.

*Exercise.* Two modes, mode 1 learned and mode 2 not ($a_1=s_1$, $a_2=0$), with
forcing $F=\frac1nr_px_p^\top$.
1. From Step 5 with $r=1$ and $B=F$, show
   $\delta W_{21}=\frac{s_1(s_1F_{21}+s_2F_{12})}{s_1^2-s_2^2}$ and $\delta W_{12}=\frac{s_1(s_1F_{12}+s_2F_{21})}{s_1^2-s_2^2}$.
2. Show that this is the fixed point of the Step 2 equations at $a_1=s_1$, $a_2=0$, and that the slowest relaxation rate there is $(s_1-s_2)/\tau$.
"""
    ),
    answer(
        r"""
**1.** With $r=1$, $j=2$, $k=1$: $\delta W_{21}=s_1A_{21}$ and $\delta W_{12}=s_1C_{21}$, which are the stated
forms.

**2.** In Step 2's equations at $a_1=s_1$, $a_2=0$ we have $d_1=\sqrt{s_1}$, $e_1=0$, $d_2=0$, $e_2=s_2$, so
$\delta W_{12}=d_1P_{12}$, $\delta W_{21}=d_1Q_{21}$, and
$\tau\dot P_{12}=s_2Q_{21}+d_1(F_{12}-d_1P_{12})$, $\tau\dot Q_{21}=d_1(F_{21}-d_1Q_{21})+s_2P_{12}$. Setting both to
zero, with $d_1^2=s_1$:

$$
s_1P_{12}-s_2Q_{21}=d_1F_{12},\qquad s_1Q_{21}-s_2P_{12}=d_1F_{21}.
$$

Multiply the first by $s_1$, the second by $s_2$, and add: $(s_1^2-s_2^2)P_{12}=d_1(s_1F_{12}+s_2F_{21})$.
Likewise $(s_1^2-s_2^2)Q_{21}=d_1(s_1F_{21}+s_2F_{12})$. Then $\delta W_{12}=d_1P_{12}$ and $\delta W_{21}=d_1Q_{21}$
give the formulas, since $d_1^2=s_1$. The matrix
$\begin{psmallmatrix}-s_1&s_2\\s_2&-s_1\end{psmallmatrix}/\tau$ has eigenvalues $(-s_1\pm s_2)/\tau$, both negative.
"""
    ),
    markdown(
        r"""
**Step 7. A sign flip.** Four items, with input coordinates (coarse, fine): dog
$(1,1)$, cat $(1,-1)$, oak $(-1,1)$, rose $(-1,-1)$. Outputs are realizable,
$y=Sx$ with $S=\mathrm{diag}(s_1,s_2)$ and $s_1>s_2$: mode 1 is animal versus plant,
mode 2 the finer split within each kind. Then $\Sigma_{xx}=I$ and $\Sigma_{yx}=S$. The
query is the cat's loss and the upweighted example is the dog.

*Exercise.*
1. Show that the endpoint influence is exactly zero.
2. On the plateau after mode 1 is learned and before mode 2 rises, show
   $$
   \mathcal I_q\approx-\frac{s_1^2s_2^2}{n(s_1^2-s_2^2)}\,(x_p)_1(x_p)_2\,(x_q)_1(x_q)_2 ,
   $$
   which for dog on cat is $+s_1^2s_2^2/\big(n(s_1^2-s_2^2)\big)$: harmful. Interpret.
3. Sketch $\mathcal I_q(t)$ from Steps 4 and 6.
"""
    ),
    answer(
        r"""
**1.** For any weights with $\Sigma_{xx}(\beta)$ invertible, the weighted least-squares solution
is $\Sigma_{yx}(\beta)\Sigma_{xx}(\beta)^{-1}=S\,\Sigma_{xx}(\beta)\Sigma_{xx}(\beta)^{-1}=S$, since every $y_j=Sx_j$. The endpoint
does not depend on $\beta$, so its derivative is zero.

**2.** On the plateau, $a_1\approx s_1$ and $a_2\approx0$, so $r_q\approx(0,\,s_2(x_q)_2)$ and
$r_p\approx(0,\,s_2(x_p)_2)$: each example's error is all fine structure. Then $F_{12}=(r_p)_1(x_p)_2/n\approx0$
and $F_{21}=s_2(x_p)_2(x_p)_1/n$. The speed channel is quiet ($\delta a_1\approx0$ after its rise,
$\delta a_2\approx0$ before its rise), so $\mathcal I_q\approx-(r_q)_2\,\delta W_{21}\,(x_q)_1$, with
$\delta W_{21}=s_1^2F_{21}/(s_1^2-s_2^2)$ from Step 6. Substituting,

$$
\mathcal I_q\approx-s_2(x_q)_2\cdot\frac{s_1^2}{s_1^2-s_2^2}\cdot\frac{s_2(x_p)_2(x_p)_1}{n}\cdot(x_q)_1 .
$$ Dog and cat
agree on the coarse coordinate and disagree on the fine one, so the product of the four
coordinates is $-1$ and $\mathcal I_q>0$.

*Interpretation.* On the plateau the network has learned only the coarse map. The dog's
remaining error is its fine output. Upweighting the dog tilts the learned coarse map so that
the shared coarse input produces some of the dog's fine output. The cat has the same coarse
input and the opposite fine output, so the tilt hurts the cat.

**3.** During mode 1's rise the speed channel dominates: strongly negative (helpful). On the
plateau the rotation channel gives the positive value above. During mode 2's rise the speed
channel is negative again and the tilt relaxes. After mode 2 is learned everything decays to
the endpoint value, zero.
"""
    ),
    markdown(
        r"""
> **Scope.** Exact linear response at fixed aligned, balanced initialization; the speed channel in closed form; the rotation channel in closed form on plateaus only. The data are whitened and realizable; with label noise the endpoint influence can be nonzero, and both channels contribute to it.

<a id="da2-dln-code"></a>

### Code 4: a dog's influence on a cat, over training

The hierarchy of Step 7 with $s_1=4$, $s_2=1$, $a_0=10^{-6}$, trained by gradient
descent with $\eta=10^{-3}$ from the fixed aligned, balanced initialization
(time is shown as $t/\tau=\eta\times$steps). We compute $\mathcal I_q(t)$ three ways:
- **finite differences** of reweighted retraining, $\beta_{\rm dog}=1\pm10^{-5}$;
- the **linear-response equations** of Step 1, integrated alongside training;
- the **closed forms**: the speed channel of Step 3 (exact), plus the plateau rotation of Step 6 (drawn only on the plateau).

Look for the sign pattern: negative through mode 1's rise, positive on the
plateau at the predicted $s_1^2s_2^2/(n(s_1^2-s_2^2))=0.267$, negative through mode
2's rise, then zero. The speed channel alone follows the full curve except
where the rotation channel is active: on the plateau and at the start of mode
2's rise, while the tilt relaxes.
"""
    ),
    code(
        r"""
t0 = time.time()
s1, s2, a0, eta, steps = 4.0, 1.0, 1e-6, 1e-3, 10000
S = np.diag([s1, s2])
Xh = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)    # dog, cat, oak, rose: (coarse, fine)
Yh = Xh @ S.T
n_h, p_, q_ = 4, 0, 1                                         # upweight the dog, query the cat
xp, xq, yq = Xh[p_], Xh[q_], Yh[q_]
t = eta * np.arange(steps + 1)


def train_dln(beta, tangent=False):
    W1 = np.sqrt(a0) * np.eye(2)
    W2 = W1.copy()
    P, Q = np.zeros((2, 2)), np.zeros((2, 2))                 # dW1, dW2 (linear response)
    Sxx, Syx = (Xh.T * beta) @ Xh / n_h, (Yh.T * beta) @ Xh / n_h
    Ws, dWs = [], []
    for _ in range(steps + 1):
        W = W2 @ W1
        E = Syx - W @ Sxx
        Ws.append(W)
        if tangent:
            dW = Q @ W1 + W2 @ P
            dWs.append(dW)
            dE = np.outer(Yh[p_] - W @ xp, xp) / n_h - dW      # Step 1: F - dW
            P, Q = P + eta * (Q.T @ E + W2.T @ dE), Q + eta * (dE @ W1.T + E @ P.T)
        W1, W2 = W1 + eta * W2.T @ E, W2 + eta * E @ W1.T
    return np.array(Ws), np.array(dWs) if tangent else None


def cat_loss(Ws):
    r = yq - Ws @ xq
    return 0.5 * (r**2).sum(1)


eps = 1e-5
bp, bm = np.ones(n_h), np.ones(n_h)
bp[p_] += eps
bm[p_] -= eps
W0, dW = train_dln(np.ones(n_h), tangent=True)
I_fd = (cat_loss(train_dln(bp)[0]) - cat_loss(train_dln(bm)[0])) / (2 * eps)
rq = yq - W0 @ xq
I_lin = -np.einsum("ti,tij,j->t", rq, dW, xq)

# closed forms (gradient flow, t in units of tau)
a = np.stack([s * np.exp(2 * s * t) / (np.exp(2 * s * t) - 1 + s / a0) for s in (s1, s2)], 1)
adot = 2 * a * (np.array([s1, s2]) - a)
da = (xp**2 / n_h) * t[:, None] * adot                        # Step 3, realizable data
rq_modes = (np.array([s1, s2]) - a) * xq
I_speed = -(rq_modes * xq * da).sum(1)
F21 = (s2 - a[:, 1]) * xp[1] * xp[0] / n_h
F12 = (s1 - a[:, 0]) * xp[0] * xp[1] / n_h
dW21 = s1 * (s1 * F21 + s2 * F12) / (s1**2 - s2**2)
dW12 = s1 * (s1 * F12 + s2 * F21) / (s1**2 - s2**2)
I_rot = -(rq_modes[:, 0] * dW12 * xq[1] + rq_modes[:, 1] * dW21 * xq[0])
t1, t2 = np.log(s1 / a0) / (2 * s1), np.log(s2 / a0) / (2 * s2)
plateau = (t > t1 + 0.8) & (t < t2 - 1.5)

print(f"mode rises at t1* = {t1:.2f}, t2* = {t2:.2f} (units of tau)")
print(f"max |finite differences - linear response| = {np.abs(I_fd - I_lin).max():.1e} (max |I| = {np.abs(I_fd).max():.2f})")
k = int(round(4.0 / eta))
print(f"plateau, t = 4: I = {I_fd[k]:+.3f}; speed channel {I_speed[k]:+.4f}, rotation formula {I_rot[k]:+.3f}, "
      f"s1^2 s2^2/(n(s1^2-s2^2)) = {s1**2 * s2**2 / (n_h * (s1**2 - s2**2)):.3f}")
for tt in [1.9, 4.0, 6.9, 10.0]:
    j = int(round(tt / eta))
    print(f"t = {tt:4.1f}: I = {I_fd[j]:+.4f}")

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
ax0.plot(t, W0[:, 0, 0] / s1, color=COL[0], label="mode 1 (animal vs plant)")
ax0.plot(t, W0[:, 1, 1] / s2, color=COL[1], label="mode 2 (within kind)")
ax0.set_ylabel(r"$a_k/s_k$")
ax0.legend(frameon=False, fontsize=8)
ax1.plot(t, I_fd, color="k", lw=2.5, label="finite differences of retraining")
ax1.plot(t, I_lin, "--", color=COL[3], label="linear-response equations (Step 1)")
ax1.plot(t, I_speed, ":", color=COL[0], label="speed channel, closed form (Step 3)")
ax1.plot(t[plateau], I_rot[plateau], color=COL[1], lw=4, alpha=0.6, label="plateau rotation (Step 6)")
ax1.axhline(0, color=GREY, lw=1)
ax1.set_yscale("symlog", linthresh=0.05)
ax1.set_xlabel(r"$t$ (units of $\tau$, i.e. $\eta\times$steps)")
ax1.set_ylabel(r"$\mathcal{I}_{\rm cat}(t)$ of the dog")
ax1.legend(frameon=False, fontsize=8, loc="upper right")
for ax in (ax0, ax1):
    for tk in (t1, t2):
        ax.axvline(tk, color=GREY, lw=0.8, ls=":")
fig.tight_layout()
plt.show()
print(f"({time.time() - t0:.1f}s)")
"""
    ),
]

# ======================================================================= stop 5
CELLS += [
    markdown(
        r"""
<a id="da2-stop-5"></a>

## 5. Learning the map: datamodels and TRAK

**Question.** Can we skip the derivative? Fix a distribution over training subsets (for
example random halves), retrain many times, and fit a predictor of $\phi$ from
the subset's indicator vector $\mathbf 1_S\in\{0,1\}^n$. A linear datamodel
$\phi(S)\approx\theta_0+\sum_i\theta_i\mathbf 1[i\in S]$ has coefficients $\theta_i$ that are
attributions *for that subset distribution*. Here $\pi(\beta)$ is retraining from
scratch on the subset with indicator $\beta$, and the datamodel is a linear fit to
$\phi(\pi(\beta))$ over random subsets.

**Cost, and the shortcut.** Datamodels needs a very large number of retrainings:
300,000 ResNet-9 models for CIFAR-10 at 50% subsets. TRAK gets close without
fitting to retraining outcomes. It linearizes the model around its trained
weights, randomly projects per-example gradients to a few thousand dimensions,
applies a one-step Gauss–Newton influence estimate in that space, and averages
over a few models trained on random subsets. Both are scored with the LDS of
[DA1 §5](da1_core.html#da1-5), the rank correlation between predicted and actual $\phi$ over random
subsets, which is the datamodel target.

**Caveat.** A high LDS at 50% subsets says how well a method predicts large
random deletions. A method can compute its local derivative exactly and still
score a low LDS at 50% subsets, and the reverse (DA1 §5).
"""
        + TRAK + EXTRAS + DATAMODELS
    ),
]

# ======================================================================= stop 6
CELLS += [
    markdown(
        r"""
<a id="da2-stop-6"></a>

## 6. Open problems and uses

Where the methods above stop working, and what attribution is used for.

- **Entangled latent concepts.** An example carries many concepts at once, and
  attribution to examples mixes them all. The question we usually care about is
  narrower: which data taught the model *this* concept? Attributing to concepts
  rather than examples is open.
- **Behaviours rather than losses.** The observable is usually a loss or a
  log-probability. Alignment questions are about behaviours, such as role-play
  or refusals, which first need a scalar $\phi$ that measures them. Grosse et al.'s
  role-play queries (§5.3.5) are one way to set this up.
- **Fine-tuning data selection.** Scores can be used to choose data instead of
  explaining it, which tests attribution by its downstream effect.
- **Unlearning.** Attribution predicts the effect of removing data; unlearning
  tries to produce that effect without retraining. Whether an IF-style update
  achieves real removal is again a question of target versus estimator.
- **Distributional attribution.** Training is random (seed, data order). A
  well-posed target averages over that randomness; Mlodozeniec et al. ([Stop 3](#da2-stop-3))
  call this distributional attribution. Single-run unrolling and fixed-seed LDS
  measure something else.
- **Scale.** Exact methods cost retraining or replay; scalable ones approximate
  curvature or sample posteriors. Validating the scalable ones at LLM scale needs
  ground truth that is itself affordable.
"""
        + EXTRAS + LESS + SURVEY
    ),
]

raise SystemExit(run(CELLS, "da2_methods", "da2_methods"))
