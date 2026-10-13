#!/usr/bin/env python3
"""Build SLT Notebook 4, "Applications" (slt4_applications): a short readings-led page.

Shape (SLT_SPECS_v1.md, 2026-10-09): one-sentence annotations, no evaluation. Groups:
stagewise development and phase transitions; susceptibilities; Bayesian influence
functions; MDL and compressibility; programs and singularities (neutral, brief);
background readings for the whole day; Watanabe's books as references only.
Every URL was fetched on 2026-10-12.
"""
from nbkit_slt import book, markdown, reading, references, run

CELLS = [
    markdown(
        r"""
# SLT Notebook 4 — Applications

**Singular Learning Theory, Iliad Intensive 2026**

*Question:* what has been done with the local learning coefficient and the
sampler of Notebook 3?

This page is a reading list, not a notebook. Each entry gets one sentence on
what it does, so that you can decide whether to open it; none is evaluated
here. The entries use the day's objects: the tempered posterior
$\pi_{n,\beta}\propto\varphi e^{-\beta nL_n}$, local expectations under it
taken with stochastic gradient Langevin dynamics (SGLD), and the local learning
coefficient $\hat\lambda$. Watanabe's two books are listed at the end as
references, not as readings.

**Route.** [4.1 Stagewise development](#slt4-s1) · [4.2 Susceptibilities](#slt4-s2) · [4.3 Bayesian influence functions](#slt4-s3) · [4.4 Compressibility](#slt4-s4) · [4.5 Programs and singularities](#slt4-s5) · [4.6 Background for the whole day](#slt4-s6)
"""
    ),
    markdown(
        r"""
<a id="slt4-s1"></a>

## 4.1 Stagewise development and phase transitions

*What this is:* Notebook 2 §2.3 said the posterior changes basin at critical sample
sizes. These papers track $\hat\lambda$ along a training run and use its
plateaus and jumps to divide training into stages.
"""
        + reading(
            "Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition",
            "https://arxiv.org/abs/2310.06301",
            "Chen, Lau, Mendel, Wei & Murfet (2023)",
            "In a small autoencoder whose critical points are regular polygons, the learning coefficients of those polygons predict the posterior's phase transitions in $n$, and the same critical points organize what SGD training does.",
            importance=4,
            length=3,
        )
        + reading(
            "Loss Landscape Degeneracy and Stagewise Development in Transformers",
            "https://arxiv.org/abs/2402.02364",
            "Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (TMLR 2025)",
            "$\\hat\\lambda$ measured through training of a small language model and an in-context regression transformer, with its changes marking stages that line up with changes in what the model computes.",
            importance=4,
            length=3,
        )
        + reading(
            "Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient",
            "https://openreview.net/forum?id=SUc1UOWndp",
            "Wang, Hoogland, van Wingerden, Furman & Murfet (ICLR 2025)",
            "The estimator restricted to a subset of the weights or of the data (the refined LLC), used to follow individual attention heads as they take on separate roles during training.",
            "Also on arXiv as [2410.02984](https://arxiv.org/abs/2410.02984).",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s2"></a>

## 4.2 Susceptibilities

*What this is:* in Notebook 1's dictionary, a derivative of a posterior expectation
with respect to a small change of the "Hamiltonian" is a susceptibility, and
it equals a posterior covariance. Here the change is a reweighting of the
data distribution.
"""
        + reading(
            "Structural Inference: Interpreting Small Language Models with Susceptibilities",
            "https://arxiv.org/abs/2504.18274",
            "Baker, Wang, Hoogland & Murfet (2025)",
            "Shifting the data distribution slightly and measuring the first-order response of a local posterior expectation, estimated with SGLD samples, gives per-component and per-token attribution scores whose low-rank structure separates functional modules of a small transformer.",
            importance=3,
            length=3,
        )
        + reading(
            "Towards Spectroscopy: Susceptibility Clusters in Language Models",
            "https://arxiv.org/abs/2601.12703",
            "Gordon, Baker, Wang, Snell, van Wingerden & Murfet (2026)",
            "Susceptibilities to upweighting a token in a context, computed as covariances under the localized tempered posterior, cluster tokens that are predicted for similar reasons; applied to a 14M-parameter model.",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s3"></a>

## 4.3 Bayesian influence functions

*What this is:* classical influence functions ask how a trained parameter
moves when one training point is reweighted, and need an inverse Hessian;
the Bayesian version asks how a posterior expectation moves, which is again
a covariance under the local posterior.
"""
        + reading(
            "Bayesian Influence Functions for Hessian-Free Data Attribution",
            "https://arxiv.org/abs/2509.26544",
            "Kreer, Wu, Adam, Furman & Hoogland (2025)",
            "Replaces the inverse-Hessian step of the classical influence function by a covariance of per-sample losses under the local tempered posterior, estimated by SGLD, so that it is defined when the Hessian is singular and scales to large models.",
            importance=3,
            length=3,
        )
        + reading(
            "The Loss Kernel: A Geometric Probe for Deep Learning Interpretability",
            "https://arxiv.org/abs/2509.26537",
            "Adam, Furman & Hoogland (2025)",
            "The covariance matrix of per-sample losses under low-loss-preserving perturbations of the parameters, used as a similarity between data points and compared with semantic hierarchies on ImageNet.",
            importance=2,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s4"></a>

## 4.4 Minimum description length and compressibility

*What this is:* Notebook 1 §1.3 read the free energy as a code length; $\lambda$ is
then the coefficient of $\log n$ in the description length of the model, and
the question is whether it predicts how well a trained network compresses.
"""
        + reading(
            "Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory",
            "https://arxiv.org/abs/2510.12077",
            "Urdshals, Lau, Hoogland, van Wingerden & Murfet (2025)",
            "Extends the minimum description length principle to models with degenerate minima and reports, across the Pythia models and several compression schemes, that $\\hat\\lambda$ tracks (sometimes linearly) how compressible a network is.",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s5"></a>

## 4.5 Programs and singularities

*What this is:* a line of work relating the structure of a Turing machine to
the local geometry of the loss of a statistical model that encodes it. Listed
for completeness; it is not on the day's path and is not assessed here.
"""
        + reading(
            "Geometry of Program Synthesis",
            "https://arxiv.org/abs/2103.16080",
            "Clift, Murfet & Wallbridge (2021)",
            "Synthesis of Turing machines set up as Bayesian inference over a smooth parameter space, so that programs appear as singularities of the loss, with small experiments.",
            importance=1,
            length=4,
        )
        + reading(
            "Programs as Singularities",
            "https://arxiv.org/abs/2504.08075",
            "Murfet & Troiani (2025)",
            "A correspondence between the structure of a Turing machine and the Taylor expansion of a negative log-likelihood at the critical point that encodes it.",
            importance=1,
            length=4,
        )
    ),
    markdown(
        r"""
<a id="slt4-s6"></a>

## 4.6 Background for the whole day

Readings that serve any of Notebook 1–Notebook 3.
"""
        + reading(
            "Keeping neural networks simple by minimizing the description length of the weights",
            "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
            "Hinton & van Camp (COLT 1993)",
            "The code-length reading of the posterior volume (Notebook 1 §1.3), from the minimum-description-length side.",
            importance=2,
            length=2,
        )
        + reading(
            "Recent Advances in Algebraic Geometry and Bayesian Statistics",
            "https://arxiv.org/abs/2211.10049",
            "Watanabe (2022)",
            "The author's survey of the theory behind Notebook 2: the free energy expansion, the poles of $\\zeta$, and the generalization error.",
            importance=3,
            length=4,
        )
        + reading(
            "A Widely Applicable Bayesian Information Criterion",
            "https://jmlr.org/papers/v14/watanabe13a.html",
            "Watanabe (JMLR 2013)",
            "The source of Notebook 3's identity $\\mathbb E_{1/\\log n}[nL_n]\\approx F_n$.",
            importance=3,
            length=3,
        )
        + reading(
            "Deep Learning is Singular, and That's Good",
            "https://arxiv.org/abs/2010.11560",
            "Wei, Murfet, Gong, Li, Gell-Redman & Quella (2020)",
            "The case that neural networks are not regular models and that this is the right starting point for a theory of them.",
            importance=3,
            length=3,
        )
        + reading(
            "Singular learning theory: exercises",
            "https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises",
            "Furman (LessWrong, 2024)",
            "A problem set on the volume computations of Notebook 2, with the same toy losses.",
            importance=3,
            length=3,
        )
        + references(
            book("Algebraic Geometry and Statistical Learning Theory", "https://doi.org/10.1017/CBO9780511800474", "Watanabe (Cambridge University Press, 2009)", "the theory of Notebook 2 with proofs"),
            book("Mathematical Theory of Bayesian Statistics", "https://doi.org/10.1201/9781315373010", "Watanabe (CRC Press, 2018)", "the regular case of Notebook 1 and the general case, with the WBIC"),
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt4", "slt4_applications"))
