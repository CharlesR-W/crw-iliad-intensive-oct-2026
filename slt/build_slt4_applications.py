#!/usr/bin/env python3
"""Build SLT Notebook 4, "Applications" (slt4_applications): a short readings-led page.

Shape (SLT_SPECS_v1.md, 2026-10-09): one-sentence annotations, no evaluation. Groups:
stagewise development and phase transitions; susceptibilities; Bayesian influence
functions; MDL and compressibility; programs and singularities (neutral, brief);
background readings for the whole day; Watanabe's books as references only.
Every URL was fetched on 2026-10-12. Figures: assets/figures/make_slt4_applications_figures.py.
The identity in 4.2 is checked in checks/slt4_applications_identities.py.
"""
from nbkit_slt import book, image, markdown, reading, references, run

CELLS = [
    markdown(
        r"""
# SLT Notebook 4 — Applications

**Singular Learning Theory, Iliad Intensive 2026**

> **Read the blog posts first.** The best introductions to this material are blog posts, and they are probably better than these notebooks: Liam Carroll's sequence [Distilling Singular Learning Theory](https://www.lesswrong.com/s/czrXjvCLsqGepybHC) (DSLT 0–4), Jesse Hoogland's [Neural networks generalize because of this one weird trick](https://www.lesswrong.com/posts/fovfuFdpuEwQzJu2w/neural-networks-generalize-because-of-this-one-weird-trick) and, with Stan van Wingerden, [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong); then Zach Furman's [Singular learning theory: exercises](https://www.lesswrong.com/posts/3HYqTAi4kD35G3BzQ/singular-learning-theory-exercises) and Olli Järviniemi's [Dialogue introduction to Singular Learning Theory](https://www.lesswrong.com/posts/CmcarN6fGgTGwGuFp/dialogue-introduction-to-singular-learning-theory). The notebooks are provided for reference: derivations, worked exercises and code to check things.

*Question:* what has been done with the local learning coefficient and the
sampler of Notebook 3?

A reading list: one sentence per paper on what it does and which idea from
Notebooks 1–3 it uses. *Notation:* the tempered posterior
$\pi_{n,\beta}\propto\varphi\,e^{-\beta nL_n}$, localised near trained weights
$w^*$; SGLD (stochastic gradient Langevin dynamics) samples it; $\hat\lambda$ is
the local learning coefficient (LLC) estimated from those samples.
"""
        + image(
            "assets/figures/slt4_applications_map.svg",
            "Where each application comes from. Sections 4.1–4.4 all compute something from SGLD samples of the local posterior (Notebook 3); 4.2 and 4.3 also extend Notebook 1 §1.1's derivative identities: the derivative of a posterior average is minus a covariance. Section 4.5 is a separate line of work on the loss geometry of Notebook 2.",
            "Diagram: Notebook 1 leads to Notebook 2 leads to Notebook 3; arrows from Notebook 3 to stagewise development, susceptibilities, Bayesian influence functions and compressibility; a dashed arrow from Notebook 2 to programs and singularities.",
        )
        + r"""
**Route.** [4.1 Stagewise development](#slt4-s1) · [4.2 Susceptibilities](#slt4-s2) · [4.3 Bayesian influence functions](#slt4-s3) · [4.4 Compressibility](#slt4-s4) · [4.5 Programs and singularities](#slt4-s5) · [4.6 Background for the whole day](#slt4-s6)
"""
    ),
    markdown(
        r"""
<a id="slt4-s1"></a>

## 4.1 Stagewise development and phase transitions

*What this is:* the posterior switches basin at a critical sample size, where
a basin with lower loss but larger $\lambda$ takes over (Notebook 2 §2.3). The
first paper compares such switches with SGD training in a toy model. The second
estimates $\hat\lambda$ at checkpoints of a training run and puts stage boundaries
where the $\hat\lambda$ curve is flat; the third follows $\hat\lambda$ of single
components across those stages.
"""
        + image(
            "assets/figures/slt4_applications_lw_free_energy_crossover.png",
            "Figure from Liam Carroll, [*DSLT 4. Phase Transitions in Neural Networks*](https://www.lesswrong.com/posts/aKBAYN5LpaQMrPqMj/dslt-4-phase-transitions-in-neural-networks) (LessWrong, 2023): the free energies $F_n(\\mathcal W_{\\pm1})$ of two regions of parameter space against $n$. The posterior concentrates on the region with the lower free energy (highlighted), and the two swap at the critical sample size $n_c$.",
            "Two free-energy curves against n cross at a critical sample size n_c; the lower curve is highlighted on each side of the crossing.",
        )
        + image(
            "assets/figures/slt4_applications_llc_curve.svg",
            "A sketch, not data, of the measurement in Hoogland et al.: $\\hat\\lambda$ at checkpoints along one training run. Stage boundaries (dashed) sit where the $\\hat\\lambda$ curve is flat; within a stage $\\hat\\lambda$ usually rises and sometimes falls, while the loss keeps decreasing.",
            "Schematic: loss falls in steps over training; the estimated LLC rises in steps, with one stage where it falls; dashed lines mark stage boundaries at the flat parts of the LLC curve.",
        )
        + reading(
            "Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition",
            "https://arxiv.org/abs/2310.06301",
            "Chen, Lau, Mendel, Wei & Murfet (2023)",
            "In a small autoencoder whose critical points are regular polygons, predicts from each polygon's loss and learning coefficient where the posterior switches basin as $n$ grows, and finds that the same critical points organise SGD training.",
            importance=4,
            length=3,
        )
        + reading(
            "Loss Landscape Degeneracy and Stagewise Development in Transformers",
            "https://arxiv.org/abs/2402.02364",
            "Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (TMLR 2025)",
            "Estimates $\\hat\\lambda$ (Notebook 3) at checkpoints of a small language model and an in-context regression transformer, and splits training into stages at the plateaus of the $\\hat\\lambda$ curve, which line up with changes in what the model computes.",
            importance=4,
            length=3,
        )
        + reading(
            "Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient",
            "https://openreview.net/forum?id=SUc1UOWndp",
            "Wang, Hoogland, van Wingerden, Furman & Murfet (ICLR 2025)",
            "Restricts the $\\hat\\lambda$ estimator to the weights of one component, or to a chosen subset of the data (the refined LLC), and uses it to follow attention heads as they take on separate roles during training.",
            "Also on arXiv as [2410.02984](https://arxiv.org/abs/2410.02984).",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s2"></a>

## 4.2 Susceptibilities

*What this is:* how much does a posterior average move when the data shift
slightly? Add a small term $h\,\Delta(w)$ to the energy $nL_n$; differentiating
as in Notebook 1 §1.1 gives, for any observable $O(w)$,

$$
\frac{\partial}{\partial h}\,\mathbb E_h[O]\Big|_{h=0}=-\beta\,\mathrm{Cov}_\beta(O,\Delta),
\qquad \mathbb E_h \text{ taken under } \propto\varphi\,e^{-\beta(nL_n+h\Delta)} .
$$

This response is a susceptibility. Shifting the data toward a distribution
with loss $L'$ gives $\Delta=n\,\Delta L$ with $\Delta L=L'-L$, and Baker et al.
normalise by $n\beta$: $\chi=-\mathrm{Cov}_\beta(O,\Delta L)$, estimated with SGLD on
the local posterior.
"""
        + reading(
            "Structural Inference: Interpreting Small Language Models with Susceptibilities",
            "https://arxiv.org/abs/2504.18274",
            "Baker, Wang, Hoogland & Murfet (2025)",
            "Computes susceptibilities for each component of a small transformer to shifts toward different kinds of data; the low-rank structure of the resulting matrix separates functional modules.",
            importance=3,
            length=3,
        )
        + reading(
            "Towards Spectroscopy: Susceptibility Clusters in Language Models",
            "https://arxiv.org/abs/2601.12703",
            "Gordon, Baker, Wang, Snell, van Wingerden & Murfet (2026)",
            "Takes the shift to be an upweighting of one token in one context, and clusters tokens by their susceptibilities, which groups tokens predicted for similar reasons; applied to a 14M-parameter model.",
            importance=3,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s3"></a>

## 4.3 Bayesian influence functions

*What this is:* which training samples does a quantity depend on? The
classical influence function needs an inverse Hessian, which does not exist when
the Hessian is degenerate. Setting $\Delta=\ell_i$, the loss on training sample
$i$, in the identity of §4.2 gives a Hessian-free version, $-\beta\,\mathrm{Cov}(O,\ell_i)$.
"""
        + reading(
            "Bayesian Influence Functions for Hessian-Free Data Attribution",
            "https://arxiv.org/abs/2509.26544",
            "Kreer, Wu, Adam, Furman & Hoogland (2025)",
            "Replaces the inverse Hessian of the classical influence function by this covariance, estimated with SGLD (Notebook 3), so that it is defined when the Hessian is singular and scales to large models.",
            importance=3,
            length=3,
        )
        + reading(
            "The Loss Kernel: A Geometric Probe for Deep Learning Interpretability",
            "https://arxiv.org/abs/2509.26537",
            "Adam, Furman & Hoogland (2025)",
            "Uses the matrix $\\mathrm{Cov}(\\ell_i,\\ell_j)$ under the local posterior as a similarity between data points, and compares it with the hierarchy of ImageNet classes.",
            importance=2,
            length=3,
        )
    ),
    markdown(
        r"""
<a id="slt4-s4"></a>

## 4.4 Minimum description length and compressibility

*What this is:* at $\beta=1$, $F_n=-\log Z_n$ is the length, in nats, of the data
encoded with the Bayesian mixture code (for continuous data, at a fixed precision), so the $\lambda\log n$ term of Notebook 2 is a
description-length cost. Does $\hat\lambda$ then predict how far a trained network
can be compressed?
"""
        + reading(
            "Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory",
            "https://arxiv.org/abs/2510.12077",
            "Urdshals, Lau, Hoogland, van Wingerden & Murfet (2025)",
            "Extends the minimum description length principle to models with degenerate minima, where $\\lambda$ sets the $\\log n$ term, and reports on the Pythia models that $\\hat\\lambda$ tracks (sometimes linearly) how compressible a network is.",
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
for completeness; it is not on the day's path.
"""
        + reading(
            "Geometry of Program Synthesis",
            "https://arxiv.org/abs/2103.16080",
            "Clift, Murfet & Wallbridge (2021)",
            "Sets up the synthesis of Turing machines as Bayesian inference over a smooth parameter space, so that programs appear as singularities of the loss, with small experiments.",
            importance=1,
            length=4,
        )
        + reading(
            "Programs as Singularities",
            "https://arxiv.org/abs/2504.08075",
            "Murfet & Troiani (2025)",
            "Sets up a correspondence between the structure of a Turing machine and the Taylor expansion of a negative log-likelihood at the critical point that encodes it.",
            importance=1,
            length=4,
        )
    ),
    markdown(
        r"""
<a id="slt4-s6"></a>

## 4.6 Background for the whole day

Readings that serve any of Notebooks 1–3.
"""
        + reading(
            "Keeping neural networks simple by minimizing the description length of the weights",
            "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
            "Hinton & van Camp (COLT 1993)",
            "Trains networks with Gaussian noise on the weights, trading prediction error against the information needed to encode the weights: the Occam factor of Notebook 1 §1.3 read as a code length.",
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
