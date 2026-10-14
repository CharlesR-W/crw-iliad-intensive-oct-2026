#!/usr/bin/env python3
"""Build slt4_applications: NB4 of the SLT day, "Applications" (readings-led, short).

Spec: SLT_SPECS_v1.md, "NB4" plus its "Background readings" and "References only"
lists. Five `##` sections (anchors slt4-1 ... slt4-5) and a closing reading list
(slt4-readings). Each section restates its question and notation in a few lines,
then lists readings with one-sentence annotations and no evaluation. No code cells.

Every URL was fetched and every annotation checked against the paper's abstract and
the cited sections; formula checks are in checks/slt4_applications_identities.py.
Review log: ../reviews/slt/slt4_applications.md.
"""
from nbkit_slt import book, markdown, reading, references, run

# ---------------------------------------------------------------- 4.1 stagewise development
CHEN = reading(
    "Dynamical versus Bayesian phase transitions in a toy model of superposition",
    "https://arxiv.org/abs/2310.06301",
    "Chen, Lau, Mendel, Wei & Murfet (2023)",
    r"In the toy model of superposition with two hidden dimensions, derives the local learning "
    r"coefficients of a family of $k$-gon solutions, checks by MCMC that the posterior moves between "
    r"them as $n$ grows as their local free energies predict (§4), and compares these transitions with "
    r"the plateaus and drops of SGD training (§5).",
    importance=3, length=3)
HOOGLAND = reading(
    "Loss landscape degeneracy and stagewise development in transformers",
    "https://arxiv.org/abs/2402.02364",
    "Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (TMLR 2025)",
    r"Estimates the LLC at checkpoints along training for a two-layer attention-only language "
    r"transformer and an in-context linear-regression transformer, splits training into stages at the "
    r"plateaus of the LLC curve (§4–5), and matches most stages to changes in internal structure or "
    r"input/output behaviour (§6–7).",
    importance=4, length=3)
REFINED = reading(
    "Differentiation and specialization of attention heads via the refined local learning coefficient",
    "https://arxiv.org/abs/2410.02984",
    "Wang, Hoogland, van Wingerden, Furman & Murfet (ICLR 2025)",
    r"Defines refined LLCs, the volume exponent with only one component's weights varying or with the "
    r"loss taken on another data distribution (§3.2), and uses them to follow how the attention heads "
    r"of a two-layer attention-only transformer differentiate during training (§4).",
    importance=3, length=3)

# ---------------------------------------------------------------- 4.2 susceptibilities
BAKER = reading(
    "Structural inference: interpreting small language models with susceptibilities",
    "https://proceedings.iclr.cc/paper_files/paper/2026/hash/57eb852970a17cf9c23d35b48fbfd191-Abstract-Conference.html",
    "Baker, Wang, Hoogland, Pathak & Murfet (ICLR 2026; also arXiv:2504.18274)",
    r"Defines the susceptibility of a component's observable to a shift in the data distribution and "
    r"estimates it with local SGLD (§2–3); the components-by-data-sources matrix of susceptibilities "
    r"then picks out modules such as induction heads in a 3M-parameter transformer (§4).",
    importance=5, length=3)
GORDON = reading(
    "Towards spectroscopy: susceptibility clusters in language models",
    "https://arxiv.org/abs/2601.12703",
    "Gordon, Baker, Wang, Snell, van Wingerden & Murfet (2026)",
    r"Upweights a single token $y$ in a context $x$, shows that the resulting susceptibilities "
    r"$\chi_{xy}$ decompose as a sum over modes of the data distribution (§2), and clusters them for "
    r"Pythia-14M into 510 clusters, about half of which match sparse-autoencoder features (§4).",
    importance=3, length=3)

# ---------------------------------------------------------------- 4.3 BIF and loss kernel
KREER = reading(
    "Bayesian influence functions for Hessian-free data attribution",
    "https://arxiv.org/abs/2509.26544",
    "Kreer, Wu, Adam, Furman & Hoogland (ICLR 2026)",
    r"Defines the BIF and its localised version, shows that under a Laplace approximation it reduces "
    r"to the classical (damped) influence function (§2, App. A), and estimates it with SGLD for "
    r"networks up to billions of parameters, testing it against retraining experiments (§3–4).",
    importance=4, length=3)
LOSSKERNEL = reading(
    "The loss kernel: a geometric probe for deep learning interpretability",
    "https://arxiv.org/abs/2509.26537",
    "Adam, Furman & Hoogland (2025)",
    r"Defines the loss kernel on the localised tempered posterior (§2), checks on a synthetic multitask "
    r"problem that it separates inputs by task (§3), and applies it to Inception-v1 on ImageNet, where "
    r"its structure follows the WordNet hierarchy (§4).",
    importance=3, length=2)

# ---------------------------------------------------------------- 4.4 MDL
URDSHALS = reading(
    "Compressibility measures complexity: minimum description length meets singular learning theory",
    "https://arxiv.org/abs/2510.12077",
    "Urdshals, Lau, Hoogland, van Wingerden & Murfet (2025)",
    r"Proves the two-part code with redundancy $\lambda\log n-(m-1)\log\log n+O_p(1)$ (§3, App. E) and, "
    r"on the Pythia suite, compares how far quantisation, factorisation and other compressions can go "
    r"before the loss rises with LLC estimates, finding close and sometimes linear correlation (§4–5).",
    importance=4, length=3)

# ---------------------------------------------------------------- 4.5 programs
CLIFT = reading(
    "Geometry of program synthesis",
    "https://arxiv.org/abs/2103.16080",
    "Clift, Murfet & Wallbridge (2021)",
    r"Proposes viewing programs as singularities of analytic varieties, or equivalently as phases of "
    r"the Bayesian posterior of a program-synthesis problem, and reports small experiments on "
    r"synthesising simple Turing machines (§6–7).",
    importance=3, length=2)
TROIANI = reading(
    "Programs as singularities",
    "https://arxiv.org/abs/2504.08075",
    "Murfet & Troiani (2025)",
    r"Embeds Turing machines in a smooth family of noisy codes and shows that two machines computing "
    r"the same function can sit at critical points with different local geometry, which the posterior "
    r"distinguishes (§1.1 for the overview, §2 for the model and its learning coefficients).",
    importance=3, length=3)

# ---------------------------------------------------------------- readings for the day
HINTON = reading(
    "Keeping neural networks simple by minimizing the description length of the weights",
    "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
    "Hinton & van Camp (COLT 1993)",
    r"*With NB1.* Adds adaptive Gaussian noise to a network's weights and minimises the expected "
    r"squared error plus the information the noisy weights carry, an early variational form of the "
    r"free energy's trade-off between fit and the volume of good parameters.",
    importance=3, length=2)
WATANABE_SURVEY = reading(
    "Recent advances in algebraic geometry and Bayesian statistics",
    "https://arxiv.org/abs/2211.10049",
    "Watanabe (Information Geometry, 2024)",
    r"*With NB1–NB2.* Watanabe's review: the Bayesian framework (§2), the asymptotic free energy and "
    r"the relation between generalisation loss, cross-validation and WAIC (§5), with the algebraic "
    r"geometry behind them in §3–4.",
    importance=4, length=3)
FURMAN = reading(
    "Singular learning theory: exercises",
    "https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises",
    "Furman (LessWrong, 2024)",
    r"*With NB1–NB2.* Exercises from reparametrised normal models and volume scaling through a "
    r"two-layer linear network, the free energy of two competing phases and the bound $\lambda\ge r/2$ "
    r"from the Fisher rank $r$, to the cumulants of $nL_n$ under the tempered posterior; starred ones "
    r"are the core.",
    importance=4, length=4)
WEI = reading(
    "Deep learning is singular, and that's good",
    "https://arxiv.org/abs/2010.11560",
    "Wei, Murfet, Gong, Li, Gell-Redman & Quella (IEEE TNNLS 2023)",
    r"*With NB2.* Argues for the learning coefficient as the complexity measure for deep networks, "
    r"comparing it as a volume dimension with effective degrees of freedom and flatness (§4) and "
    r"relating it to generalisation error in experiments such as reduced-rank regression (§5).",
    importance=3, length=3)
WBIC = reading(
    "A widely applicable Bayesian information criterion",
    "https://jmlr.org/papers/v14/watanabe13a.html",
    "Watanabe (JMLR 2013)",
    r"*With NB3.* Defines WBIC, the posterior mean of $nL_n$ at inverse temperature $\beta=1/\log n$, "
    r"and proves, under conditions that cover singular and some unrealisable models, that it shares the "
    r"leading terms $nL_n(w_0)+\lambda\log n$ of the free energy $F_n$ (Theorem 4).",
    importance=4, length=3)


INDEX = r"""
# SLT 4: Applications

**Singular learning theory day, Iliad Intensive 2026**

What are the quantities from NB1–NB3 (the free energy $F_n=-\log Z_n$, the learning
coefficient $\lambda$, covariances under the tempered posterior) used for on real
networks? This page is a reading list with a few lines of set-up per section, and two
short identities in §4.2–4.3; there is no code and there are no exercises, and nothing in
NB1–NB3 depends on it. Keystone: Baker et al. (§4.2), which defines the susceptibility as
a posterior covariance; §4.3 applies the same identity to single training samples.

- 4.1 [Stagewise development and phase transitions](#slt4-1): Hoogland et al.
- 4.2 [Susceptibilities](#slt4-2): Baker et al.
- 4.3 [Bayesian influence functions and the loss kernel](#slt4-3): Kreer et al.
- 4.4 [MDL and compressibility](#slt4-4): Urdshals et al.
- 4.5 [Programs and singularities](#slt4-5): Clift, Murfet & Wallbridge; Murfet & Troiani.
- [Readings for the whole day](#slt4-readings), tagged by notebook.
"""

CELLS = [
    markdown(INDEX),
    markdown(
        r"""
<a id="slt4-1"></a>

## 4.1 Stagewise development and phase transitions

As $n$ grows, which region of parameter space does the posterior favour? And separately,
what does the learning coefficient show when it is measured at checkpoints along a
training run? For a region $W_\alpha$ of parameter space, the local free energy at
$\beta=1$ is $F_n(W_\alpha)=-\log\int_{W_\alpha}\varphi(w)e^{-nL_n(w)}dw\approx
nL_n(w_\alpha)+\lambda_\alpha\log n$, where $w_\alpha$ minimises the population loss $L$
on $W_\alpha$ and $\lambda_\alpha$ is the volume exponent there. The posterior concentrates
on the region with the smallest $F_n(W_\alpha)$, so a region with lower loss but larger
$\lambda$ can take over as $n$ grows (NB2 §3). "Phase transition" here means that the
dominant region changes at some critical $n$; with $d$ finite and no thermodynamic limit
the switch is a smooth crossover, not a singularity of $F_n$. The local learning
coefficient (LLC) $\lambda(w^*)$ is the volume exponent of the basin around one
parameter $w^*$, estimated with stochastic-gradient Langevin dynamics (SGLD) as in NB3
§3. Measuring it along a training run treats each checkpoint $w^*$ as the centre of its
own local posterior.
"""
        + HOOGLAND + CHEN + REFINED
    ),
    markdown(
        r"""
<a id="slt4-2"></a>

## 4.2 Susceptibilities

How does a posterior expectation respond when the data distribution is nudged? Mix the
true distribution $q$ with another, $q_h=(1-h)q+hq'$, so that the population loss becomes
$L_h=(1-h)L+hL'$, and take the tempered posterior
$p_{\beta,h}(w)\propto\varphi(w)e^{-n\beta L_h(w)}$, with prior $\varphi$ (in practice
$L_h$ is replaced by its empirical estimate). Pick an observable $O(w)$, for example the
loss restricted to one attention head, with posterior mean $\langle O\rangle_{\beta,h}$.
The susceptibility is

$$\chi=\frac{1}{n\beta}\,\frac{\partial\langle O\rangle_{\beta,h}}{\partial h}\Big|_{h=0}
=-\mathrm{Cov}_\beta\big(O,\,\Delta L\big),\qquad \Delta L=L'-L.$$

The second equality only needs differentiating under the integral. It is the
fluctuation–response identity behind the magnetic susceptibility in statistical
mechanics, and here it holds exactly at finite $n$ and $d$, with no thermodynamic limit.
In practice the prior is replaced by a Gaussian centred at the trained weights $w^*$
(localisation, as for the LLC) and the covariance is estimated from SGLD samples.
"""
        + BAKER + GORDON
    ),
    markdown(
        r"""
<a id="slt4-3"></a>

## 4.3 Bayesian influence functions and the loss kernel

Which training samples does a quantity depend on, and which samples does the model treat
alike? The classical influence function needs the inverse Hessian at the trained
weights, which does not exist when the Hessian is degenerate. Instead give each training
loss $\ell_i(w)=-\log p(z_i\mid w)$, with $nL_n=\sum_i\ell_i$, its own inverse
temperature $\beta_i$ in the posterior,
$p(w\mid D_n)\propto\varphi(w)\exp\!\big(-\sum_i\beta_i\ell_i(w)\big)$, so that all
$\beta_i=1$ is Bayes. The identity of §4.2, with one sample as the perturbation, gives
the Bayesian influence function (BIF) of sample $z_i$ on an observable $O(w)$:

$$\mathrm{BIF}(z_i,O)=\frac{\partial\langle O\rangle}{\partial\beta_i}\Big|_{\beta_i=1}
=-\mathrm{Cov}\big(\ell_i,O\big).$$

The loss kernel is the covariance matrix
$\mathcal K(z_i,z_j)=\mathrm{Cov}(\ell_i,\ell_j)=-\mathrm{BIF}(z_i,\ell_j)$, which is
positive semi-definite. Both papers use the localised posterior: the prior is replaced by
a Gaussian of precision $\gamma$ centred at $w^*$, and the covariances are estimated from
SGLD samples.
"""
        + KREER + LOSSKERNEL
    ),
    markdown(
        r"""
<a id="slt4-4"></a>

## 4.4 MDL and compressibility

How many nats does it cost to describe a model good enough for the data? A two-part code
sends a parameter and then the data encoded with it. Its redundancy $R_n$ is the excess
code length over the code built from the true distribution $q$. Let
$K(w)=D_{\rm KL}(q\,\|\,p(\cdot\mid w))$, which is $L(w)-L(w_0)$ when $q$ is realisable,
and let $\lambda$ and $m$ be the exponent and multiplicity of the volume of $\{K<t\}$,
$V(t)\sim c\,t^{\lambda}(-\log t)^{m-1}$ as $t\to0$. For a realisable model on a finite
alphabet, with every probability bounded below and the analytic conditions of their
App. E.1, Urdshals et al. construct a two-part code with

$$R_n=\lambda\log n-(m-1)\log\log n+O_p(1),$$

the same leading terms as the averaged free energy $\bar F_n$ in NB2 §1. A realisable
regular model has $\lambda=d/2$ and $m=1$, which gives the BIC term
$\tfrac d2\log n$ of NB1 §3.
"""
        + URDSHALS
    ),
    markdown(
        r"""
<a id="slt4-5"></a>

## 4.5 Programs and singularities

Can a discrete program be treated as a point in a continuous parameter space, so that the
volume exponent says something about the program? These papers embed Turing machine codes
in a smooth space of noisy codes $w$, with a negative log-likelihood $L(w)$ that has the
machines among its critical points. The learning coefficient $\lambda$ is then the volume
exponent of the basin around a machine, and the posterior weighs machines by loss and
$\lambda$ as in §4.1.
"""
        + CLIFT + TROIANI
    ),
    markdown(
        r"""
<a id="slt4-readings"></a>

## Readings for the whole day

These go with NB1–NB3, as tagged. Notation as on the other pages: $F_n=-\log Z_n$ is the
free energy, $L_n$ the empirical negative log-likelihood, $w_0$ an optimal parameter,
$\lambda$ the learning coefficient.
"""
        + HINTON + WATANABE_SURVEY + FURMAN + WEI + WBIC
        + references(
            book("Algebraic Geometry and Statistical Learning Theory",
                 "https://doi.org/10.1017/CBO9780511800474", "Watanabe (Cambridge University Press, 2009)"),
            book("Mathematical Theory of Bayesian Statistics",
                 "https://doi.org/10.1201/9781315373010", "Watanabe (CRC Press, 2018)"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt4_applications", "slt4_applications"))
