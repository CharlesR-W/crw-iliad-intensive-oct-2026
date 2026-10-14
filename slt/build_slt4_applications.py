#!/usr/bin/env python3
"""Build SLT Notebook 4, "Applications" (slt4_applications): a short catalogue of papers.

Shape (SLT_SPECS_v1.md; streamlined 2026-10-14): one line per paper, no evaluation. Groups:
stagewise development and phase transitions; susceptibilities; Bayesian influence
functions; MDL and compressibility; programs and singularities (neutral, brief).
Every URL was fetched on 2026-10-12 or 2026-10-14. Figures: assets/figures/make_slt4_applications_figures.py.
The identity in 4.2 is checked in checks/slt4_applications_identities.py.
"""
from nbkit_slt import image, markdown, reading, run

DSLT4 = "https://www.lesswrong.com/posts/aKBAYN5LpaQMrPqMj/dslt-4-phase-transitions-in-neural-networks"

CELLS = [
    markdown(
        r"""
# SLT Notebook 4 — Applications

**Singular Learning Theory, Iliad Intensive 2026**

> **These notebooks are a backup.** Read the blog posts first: Liam Carroll's [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises). Come here for a derivation, a worked exercise, or code to check something. The original Iliad worksheet, *Singular Learning Theory* by Kai Ogden, Matthew Farrugia-Roberts and Zach Furman, is also here: [with solutions](_sources/pdf/singular-learning-theory.pdf), [without](_sources/pdf/singular-learning-theory-nosol.pdf), and Furman's [slides](_sources/slides/slt-slides.pdf).

*Question:* what has been done with the local learning coefficient and the
sampler of Notebook 3?

*Notation:* $\hat\lambda$
is the local learning coefficient (LLC), estimated from SGLD samples of the
tempered posterior near the trained weights (Notebook 3).
"""
        + image(
            "assets/figures/slt4_applications_map.svg",
            "Sections 4.1–4.4 each compute something from SGLD samples of the local posterior (Notebook 3); 4.5 is a separate line of work on loss geometry.",
            "Diagram: Notebook 1 leads to Notebook 2 leads to Notebook 3; arrows from Notebook 3 to stagewise development, susceptibilities, Bayesian influence functions and compressibility; a dashed arrow from Notebook 2 to programs and singularities.",
        )
        + r"""
**Route.** [4.1 Stagewise development](#slt4-s1) · [4.2 Susceptibilities](#slt4-s2) · [4.3 Bayesian influence functions](#slt4-s3) · [4.4 Compressibility](#slt4-s4) · [4.5 Programs and singularities](#slt4-s5)
"""
    ),
    markdown(
        r"""
<a id="slt4-s1"></a>

## 4.1 Stagewise development and phase transitions

*What this is:* the posterior switches basin at a critical sample size, where a
basin with lower loss but a larger learning coefficient takes over (Notebook 2
§2.3). These papers look for such switches, and for stages in training.
"""
        + image(
            "assets/figures/slt4_applications_lw_free_energy_crossover.png",
            f"Figure from Liam Carroll, [*DSLT 4. Phase Transitions in Neural Networks*]({DSLT4}) (LessWrong, 2023): the free energies of two regions cross at a critical sample size $n_c$, and the posterior moves to the lower one.",
            "Two free-energy curves against n cross at a critical sample size n_c; the lower curve is highlighted on each side of the crossing.",
        )
        + image(
            "assets/figures/slt4_applications_llc_curve.svg",
            "A sketch, not data: the LLC estimated along one training run, with stage boundaries (dashed) where its curve is flat.",
            "Schematic: test loss falls in steps over training; the estimated LLC rises in steps, with one stage where it falls; dashed lines mark stage boundaries at the flat parts of the LLC curve.",
        )
        + reading(
            "Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition",
            "https://arxiv.org/abs/2310.06301",
            "Chen, Lau, Mendel, Wei & Murfet (2023)",
            "In a toy autoencoder, predicts the posterior's basin switches from the loss and learning coefficient of its polygon-shaped critical points, and finds that the same critical points organise SGD training.",
            importance=4,
            length=3,
        )
        + reading(
            "Loss Landscape Degeneracy and Stagewise Development in Transformers",
            "https://arxiv.org/abs/2402.02364",
            "Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (TMLR 2025)",
            "Estimates the LLC at checkpoints of two small transformers and splits training into stages at its plateaus, which line up with changes in what the model computes.",
            importance=4,
            length=3,
        )
        + reading(
            "Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient",
            "https://openreview.net/forum?id=SUc1UOWndp",
            "Wang, Hoogland, van Wingerden, Furman & Murfet (ICLR 2025)",
            "Restricts the LLC to one component's weights or to a subset of the data, and follows attention heads as they specialise (also [arXiv:2410.02984](https://arxiv.org/abs/2410.02984)).",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s2"></a>

## 4.2 Susceptibilities

*What this is:* how much does a posterior average move when the data shift
slightly? Add a small term $h\Delta$ to the energy $nL_n$ and differentiate, as in
Notebook 1 §1.1. For any observable $O$ the answer is a covariance:

$$
\frac{\partial}{\partial h}\,\mathbb E_h[O]\Big|_{h=0}=-\beta\,\mathrm{Cov}_\beta(O,\Delta),
\qquad \mathbb E_h \text{ under } \propto\varphi\,e^{-\beta(nL_n+h\Delta)} ,
$$

with $\beta$ the inverse temperature and $\varphi$ the prior, both held fixed. This
response is a susceptibility. For a shift in the data, the perturbation is $n$
times the change in average loss; Baker et al. divide the derivative by $n\beta$,
which leaves minus the covariance with that change. It is estimated with SGLD.
"""
        + reading(
            "Structural Inference: Interpreting Small Language Models with Susceptibilities",
            "https://arxiv.org/abs/2504.18274",
            "Baker, Wang, Hoogland & Murfet (2025)",
            "Measures the susceptibility of each component of a small transformer to shifts toward different kinds of data; the low-rank structure of the result separates functional modules.",
            importance=3,
            length=3,
        )
        + reading(
            "Towards Spectroscopy: Susceptibility Clusters in Language Models",
            "https://arxiv.org/abs/2601.12703",
            "Gordon, Baker, Wang, Snell, van Wingerden & Murfet (2026)",
            "Upweights one token in one context and clusters tokens by their susceptibilities, grouping tokens predicted for similar reasons, in a 14M-parameter model.",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s3"></a>

## 4.3 Bayesian influence functions

*What this is:* which training samples does a quantity depend on? The classical
influence function needs an inverse Hessian, which does not exist when the
Hessian is degenerate. Taking the perturbation in §4.2 to be the loss on one
training sample gives a covariance, with no Hessian.
"""
        + reading(
            "Bayesian Influence Functions for Hessian-Free Data Attribution",
            "https://arxiv.org/abs/2509.26544",
            "Kreer, Wu, Adam, Furman & Hoogland (2025)",
            "Defines this Bayesian influence function, estimates it with SGLD, and scales it to large models.",
            importance=3,
            length=3,
        )
        + reading(
            "The Loss Kernel: A Geometric Probe for Deep Learning Interpretability",
            "https://arxiv.org/abs/2509.26537",
            "Adam, Furman & Hoogland (2025)",
            "Uses the posterior covariance of per-sample losses as a similarity between data points, and compares it with the ImageNet class hierarchy.",
            importance=2,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s4"></a>

## 4.4 Minimum description length and compressibility

*What this is:* at $\beta=1$ the free energy is the length of the data under the
Bayesian mixture code (for continuous data, up to a fixed-precision term), so the $\lambda\log n$ term of Notebook 2 is a
description-length cost. Does the LLC predict how far a trained network can be
compressed?
"""
        + reading(
            "Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory",
            "https://arxiv.org/abs/2510.12077",
            "Urdshals, Lau, Hoogland, van Wingerden & Murfet (2025)",
            "Extends minimum description length to models with degenerate minima and reports, on the Pythia models, that the LLC tracks (sometimes linearly) how compressible a network is.",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s5"></a>

## 4.5 Programs and singularities

*What this is:* a line of work relating the structure of a Turing machine to the
loss geometry of a model that encodes it.
"""
        + reading(
            "Geometry of Program Synthesis",
            "https://arxiv.org/abs/2103.16080",
            "Clift, Murfet & Wallbridge (2021)",
            "Sets up the synthesis of Turing machines as Bayesian inference, so that programs appear as singularities of the loss.",
            importance=1,
            length=4,
        )
        + reading(
            "Programs as Singularities",
            "https://arxiv.org/abs/2504.08075",
            "Murfet & Troiani (2025)",
            "Relates the structure of a Turing machine to the Taylor expansion of the loss at the critical point that encodes it.",
            importance=1,
            length=4,
        )
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "slt4", "slt4_applications"))
