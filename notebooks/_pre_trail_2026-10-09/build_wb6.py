#!/usr/bin/env python3
"""Build WB6, "Mysteries myriad": a reading catalogue of strange empirical facts about deep learning.

An intro with a route list (one line per section), then 13 sections, anchors
`wb6-section-1` ... `wb6-section-13`. Each section has a heading, at most one
line naming related sections in other workbooks, and one to four `reading()`
blurbs of 1-3 sentences each saying what the paper shows (§5 and §13 also have a short framing paragraph). Four sections also carry
short derivation exercises with worked answers: §5 (packing almost-orthogonal
directions), §9 (a quadratic toy for gradient overlap), §10 (scale invariance
under batch norm), §11 (sparse vs low-rank updates; what "unchanged" means in
bfloat16). Those five answers are folded under "Solution" in the single HTML
notebook. §5's superposition material and §13 moved here from WB5 on 2026-10-09
(pre-move builders in _old_wb5/).
Editorial record: reviews/wb6-editorial.md.
"""
from nbkit import answer, figure, markdown, reading, run

SP = "assets/interactive/superposition.html"

SECTIONS = [
    ("1", "Transformer middle layers",
     "intermediate layers often give better embeddings than the last layer."),
    ("2", "Bottom-up learning",
     "lower layers settle first during training."),
    ("3", "Model stitching works",
     "one linear layer joins the bottom of one network to the top of another."),
    ("4", "Critical periods",
     "a deficit early in training can persist through later training."),
    ("5", "Superposition and adversarial attacks (and on you too)",
     "more sparse features than dimensions; predictive but non-robust features; perturbations that bias briefly shown humans. One exercise, one interactive figure."),
    ("6", "Critical modules",
     "many layers can be reset to their initial weights after training; a few cannot."),
    ("7", "Distillation and pruning",
     "a small model learns much more from a large model's soft outputs than from labels; pruning is in §8."),
    ("8", "Rewinding and lottery tickets",
     "sparse trainable subnetworks, and stability to SGD noise early in training."),
    ("9", "Gradient descent in a tiny subspace",
     "the gradient concentrates in a few top Hessian directions. One exercise."),
    ("10", "Spectra and curvature",
     "the Hessian as a bulk plus outliers, and what batch norm does. One exercise."),
    ("11", "Low-rank RL updates and the KL leash",
     "RL updates reported as sparse yet full-rank, or as rank-1 dominated; on-policy RL forgets less. Two exercises."),
    ("12", "Physics of language models",
     "transformers trained on synthetic grammars, probed for parse structure."),
    ("13", "Hidden signals: low-rank logits and subliminal learning",
     "models' outputs share low-rank structure; traits pass from teacher to student through data that never mentions them."),
]

TITLES = {k: t for k, t, _ in SECTIONS}
ROUTE = "\n".join(f"- [§{k}. {t}](#wb6-section-{k}): {p}" for k, t, p in SECTIONS)


def head(k: str, links: str = "") -> str:
    out = f'<a id="wb6-section-{k}"></a>\n\n## {k}. {TITLES[k]}'
    if links:
        out += f"\n\n*Connects to:* {links}"
    return out


CELLS = [
    markdown(
        r"""
# Workbook 6 — Mysteries myriad

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

A catalogue of readings on strange empirical facts about trained networks. Each section links one to four papers, with a few sentences on what each shows, so you can decide what to read. The sections are independent. §5 and §9–§11 also have short exercises. Worked answers to those exercises are folded under "Solution" directly after each one; they are available to everyone.

""" + ROUTE + r"""

The Training Dynamics materials take up the dynamics behind §4 and §8–§11.
"""
    ),

    # ------------------------------------------------------------------ 1
    markdown(
        head("1", "WB2 §1.F, WB5 §3.B.") + "\n\n" + reading(
            "Layer by layer: uncovering hidden representations in language models",
            "https://arxiv.org/abs/2502.02013",
            "Skean, Arefin, Zhao, Patel, Naghiyev, LeCun & Shwartz-Ziv (ICML 2025)",
            "On 32 text-embedding tasks, intermediate layers of the tested models (transformers and state-space models) often give better downstream features than the final layer. "
            "The paper proposes label-free measures of representation quality, based on information theory, geometry and invariance to input perturbations, and uses them to track how this changes with depth.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 2
    markdown(
        head("2", "WB5 §3.A.") + "\n\n" + reading(
            "SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability",
            "https://arxiv.org/abs/1706.05806",
            "Raghu, Gilmer, Yosinski & Sohl-Dickstein (NeurIPS 2017)",
            "Introduces SVCCA, an SVD truncation followed by canonical correlation analysis, for comparing layer representations. "
            "With it the authors find that layers have low effective dimensionality and that networks converge from the bottom up during training, which motivates \"freeze training\": freezing lower layers in sequence to save computation.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 3
    markdown(
        head("3", "WB5 §3.A.") + "\n\n" + reading(
            "Revisiting model stitching to compare neural representations",
            "https://arxiv.org/abs/2106.07682",
            "Bansal, Nakkiran & Barak (NeurIPS 2021)",
            "Joins the lower layers of one trained network to the upper layers of another through one trainable linear layer (a 1×1 convolution for CNNs) and measures the loss in performance. "
            "Networks of the same architecture trained with different seeds, or with supervised versus self-supervised objectives, stitch with little loss. "
            "Representations learned with more data, width or training time can be stitched into weaker models and improve them (\"more is better\").",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 4
    markdown(
        head("4", "[§8](#wb6-section-8) (what is decided early).") + "\n\n" + reading(
            "Critical learning periods in deep neural networks",
            "https://arxiv.org/abs/1711.08856",
            "Achille, Rovere & Soatto (ICLR 2019)",
            "In their experiments, training on blurred images for the first epochs left a deficit that persisted through subsequent training on clean images, while the same deficit introduced later did little harm. "
            "Deficits that leave low-level statistics intact, such as vertical flips, had no lasting effect. "
            "The Fisher information in the weights rises early in training and then falls, which the authors relate to the critical period.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 5
    markdown(
        head("5", "WB5 §3.C (linear representations).") + r"""

High-dimensional space holds exponentially many almost-orthogonal directions, so a network can store more sparse features than it has dimensions, at the price of interference when several are active at once. Interference is one proposed account of adversarial vulnerability: a small input change aligned with many stored directions moves many features at once. Ilyas et al. below give a competing account, in which adversarial perturbations exploit features that really are predictive. Storing a feature redundantly is a candidate route to error correction, for which there is no reading here.

""" + reading(
            "Toy models of superposition",
            "https://transformer-circuits.pub/2022/toy_model/index.html",
            "Elhage et al. (Anthropic, 2022; also arXiv 2209.10652)",
            "In a toy ReLU autoencoder, networks store more sparse features than they have dimensions, which offers an explanation of polysemantic neurons. "
            "Whether they do so is a phase change in sparsity and importance, the stored features form regular polytopes, and in these toy models superposition goes together with adversarial vulnerability. "
            "The section 'Superposition as a Phase Change' has the two-feature phase diagram in the figure below.",
            importance=5, length=4,
        ) + "\n\n" + reading(
            "Adversarial examples are not bugs, they are features",
            "https://arxiv.org/abs/1905.02175",
            "Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry (NeurIPS 2019)",
            "Argues that adversarial vulnerability comes from non-robust features: patterns that really do predict the label but are brittle and incomprehensible to humans. "
            "Training on adversarially perturbed images relabelled with the attack's target class (a set that looks mislabelled to a human) still gives non-trivial accuracy on the clean test set. "
            "Conversely, a dataset restricted to robust features gives a model with non-trivial robustness under standard training.",
            importance=4, length=3,
        ) + "\n\n" + reading(
            "Adversarial examples that fool both computer vision and time-limited humans",
            "https://arxiv.org/abs/1802.08195",
            "Elsayed, Shankar, Cheung, Papernot, Kurakin, Goodfellow & Sohl-Dickstein (NeurIPS 2018)",
            "Builds perturbations that transfer across an ensemble of models with a retina-like front end, then shows them to people for a fraction of a second. "
            "The perturbations measurably bias the humans' choice between two classes.",
            importance=2, length=2,
        ) + r"""

**Exercise 5.1 (packing almost-orthogonal directions).** Let $u,v$ be independent, uniformly random unit vectors in $\mathbb R^m$.

- (a) Show that $\mathbb E\langle u,v\rangle=0$ and $\mathbb E\langle u,v\rangle^2=1/m$.
- (b) Take as given the concentration bound $\Pr(|\langle u,v\rangle|>\varepsilon)\le2e^{-m\varepsilon^2/2}$. Show that if $N^2e^{-m\varepsilon^2/2}<1$, then $N$ independent random unit vectors have all pairwise $|\cos|\le\varepsilon$ with positive probability. Conclude that $\mathbb R^m$ contains about $e^{m\varepsilon^2/4}$ such directions.

*Hint:* (a) condition on $v$ and use $\mathbb E[uu^\top]=I/m$ (isotropy, and the trace is 1); (b) union bound over pairs.
"""
    ),
    answer(
        r"""
**(a)** $u$ and $-u$ have the same distribution, so $\mathbb E\langle u,v\rangle=0$. By isotropy $\mathbb E[uu^\top]=\kappa I$, and taking the trace gives $\mathbb E\|u\|^2=1=\kappa m$. Conditioning on $v$, $\mathbb E\langle u,v\rangle^2=v^\top\mathbb E[uu^\top]v=1/m$. A typical overlap is of size $1/\sqrt m$.

**(b)** There are $N(N-1)/2$ pairs. By the union bound,
$$\Pr(\text{some pair has }|\cos|>\varepsilon)\le\frac{N(N-1)}2\cdot2e^{-m\varepsilon^2/2}<N^2e^{-m\varepsilon^2/2}<1,$$
so with positive probability no pair is bad, and such a configuration exists. The condition holds for any $N<e^{m\varepsilon^2/4}$: exponentially many directions in $m$, against only $m$ exactly orthogonal ones. If features are sparse, the small overlaps rarely matter at the same time, which is why storing more than $m$ of them can pay.
"""
    ),
    markdown(
        r"""
The smallest case already shows the trade-off. The toy model compresses features $x$ to $Wx$ and reads them back as $\mathrm{ReLU}(W^\top Wx+b)$, with squared error weighted by each feature's importance. Put two features with importances $1$ and $r\le1$, each active with probability $p$, into one dimension. Dedicating the dimension to feature 1 (with the best constant guess for feature 2) costs $r$ times the variance of feature 2, which is of order $rp$. Storing both antipodally ($W=[1,-1]$, with the ReLU erasing negative interference) costs only when both fire, with probability $p^2$. With features uniform on $[0,1]$ when active, antipodal storage wins for $p<4r/(2+5r)$: sparse features get superposed even when one matters little, and dense features force a choice.

""" + figure(SP, r"Panel 1 is the two-feature, one-dimension phase diagram over density and relative importance: the dashed boundary is the closed-form comparison above, the solid one optimizes weights and biases, and dots are trained models. Panel 2 trains $n=5$ or $6$ features in $m=2$ dimensions in your browser; as sparsity increases, the features go from dedicated dimensions to antipodal pairs to a regular polygon.")
    ),

    # ------------------------------------------------------------------ 6
    markdown(
        head("6", "WB2 §4.C.") + "\n\n" + reading(
            "Are all layers created equal?",
            "https://arxiv.org/abs/1902.01996",
            "Zhang, Bengio & Singer (2019; JMLR 2022)",
            "Resets one layer of a trained network at a time, either to its initial weights or to fresh random weights. "
            "For many layers, resetting to the initial weights barely changes performance (\"robust\" layers); a few layers are \"critical\", and which ones depends on the architecture. "
            "The authors discuss what this implies for generalization measures and caution that architecture matters.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 7
    markdown(
        head("7", "[§8](#wb6-section-8) for pruning.") + "\n\n" + reading(
            "Distilling the knowledge in a neural network",
            "https://arxiv.org/abs/1503.02531",
            "Hinton, Vinyals & Dean (2015)",
            "Training a model on the temperature-softened outputs of a larger model (MNIST) or of an ensemble (speech recognition) transfers much of the teacher's performance; the paper argues that the small probabilities given to wrong classes carry much of the useful information. "
            "In one MNIST experiment the digit 3 never appears in the transfer set, yet the student gets most test 3s right, and 98.6% once its bias for that class is raised.",
            importance=5, length=2,
        )
    ),

    # ------------------------------------------------------------------ 8
    markdown(
        head("8", "WB5 §3.A (parameter symmetries).") + "\n\n" + reading(
            "The lottery ticket hypothesis: finding sparse, trainable neural networks",
            "https://arxiv.org/abs/1803.03635",
            "Frankle & Carbin (ICLR 2019)",
            "Iterative magnitude pruning, with the surviving weights reset to their initial values, finds subnetworks under 10–20% of the original size that train in isolation to the original accuracy on MNIST and CIFAR-10. "
            "The same masks with re-randomized weights do worse. "
            "Deeper networks (VGG, ResNet) needed learning-rate warmup or a lower learning rate.",
            importance=5, length=3,
        ) + "\n\n" + reading(
            "Linear mode connectivity and the lottery ticket hypothesis",
            "https://arxiv.org/abs/1912.05671",
            "Frankle, Dziugaite, Roy & Carbin (ICML 2020)",
            "Calls a network stable to SGD noise from step $k$ if two copies trained on from step $k$ with different data orders are linearly connected: error does not rise along the straight line between them. "
            "Subnetworks found by iterative magnitude pruning reach full accuracy only when they are stable in this sense. In the tested settings that happens at initialization for small MNIST networks, but only early in training for larger ones such as ResNet-50 on ImageNet.",
            importance=4, length=3,
        )
    ),

    # ------------------------------------------------------------------ 9
    markdown(
        head("9", "WB2 §4.B.") + "\n\n" + reading(
            "Gradient descent happens in a tiny subspace",
            "https://arxiv.org/abs/1812.04754",
            "Gur-Ari, Roberts & Dyer (2018)",
            "Measures how much of the gradient lies in the span of the top Hessian eigenvectors during training. "
            "After a short initial phase most of it does; the number of top directions matches the number of classes in the tested classifiers, and the subspace changes slowly. "
            "Short, with a toy model explaining the count.",
            importance=5, length=2,
        ) + r"""

**Exercise 9.1 (a quadratic toy).** This toy is not the paper's model. It shows that "the gradient lies in the top subspace" and "the loss goes down along the top subspace" can come apart. Take $L(\theta)=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$ with $H$ symmetric positive semi-definite, orthonormal eigenvectors $v_i$ with eigenvalues $\lambda_1\ge\lambda_2\ge\dots$, and write $\theta-\theta^*=\sum_i c_iv_i$. The paper's overlap is $\lVert P_kg\rVert^2/\lVert g\rVert^2$, where $g=\nabla L$ and $P_k$ projects onto $v_1,\dots,v_k$.

- (a) Show that the overlap equals $\sum_{i\le k}\lambda_i^2c_i^2\big/\sum_i\lambda_i^2c_i^2$, and that $L=\tfrac12\sum_i\lambda_ic_i^2$.
- (b) For gradient descent $\theta\leftarrow\theta-\eta g$, show that $c_i(t)=(1-\eta\lambda_i)^t\,c_i(0)$.
- (c) Take one outlier $\lambda_1=\Lambda$ and $p-1$ bulk directions with $\lambda_i=\mu$, where $0<\mu\le\Lambda/2$, and $k=1$. Assume $c_1(0)\neq0$ and some bulk $c_i(0)\neq0$. For which step sizes $0<\eta<2/\Lambda$ does the overlap tend to 1 as $t\to\infty$?
- (d) Take $\Lambda=100$, $\mu=1$, $p=1000$, $\eta=0.0199$ and all $c_i(0)=1$. At $t=0$ and at $t=100$, compute the overlap and the fraction of the one-step loss decrease that comes from the bulk.

*Hint for (d):* direction $i$ contributes $\tfrac12\lambda_ic_i^2\bigl(1-(1-\eta\lambda_i)^2\bigr)$ to the decrease in one step.
"""
    ),
    answer(
        r"""
**(a)** $g=H(\theta-\theta^*)=\sum_i\lambda_ic_iv_i$ and $P_kg=\sum_{i\le k}\lambda_ic_iv_i$. By orthonormality, $\lVert g\rVert^2=\sum_i\lambda_i^2c_i^2$ and $\lVert P_kg\rVert^2=\sum_{i\le k}\lambda_i^2c_i^2$. Likewise $L=\tfrac12\sum_i\lambda_ic_i^2$.

**(b)** $\theta-\theta^*\leftarrow(\theta-\theta^*)-\eta H(\theta-\theta^*)=(I-\eta H)(\theta-\theta^*)$, and $v_i$ is an eigenvector of $I-\eta H$ with eigenvalue $1-\eta\lambda_i$. So $c_i\leftarrow(1-\eta\lambda_i)c_i$ at each step.

**(c)** Let $r_1=|1-\eta\Lambda|$ and $r_b=1-\eta\mu$. Since $\eta<2/\Lambda$ and $\mu\le\Lambda/2$, $0<\eta\mu<1$, so $0<r_b<1$. By (a) and (b), with $k=1$,
$$\text{overlap}(t)=\frac{\Lambda^2c_1(0)^2\,r_1^{2t}}{\Lambda^2c_1(0)^2\,r_1^{2t}+\mu^2\sum_{i\ge2}c_i(0)^2\,r_b^{2t}},$$
which tends to 1 if $r_1>r_b$ and to 0 if $r_1<r_b$. If $\eta\Lambda\le1$, then $r_1=1-\eta\Lambda<1-\eta\mu=r_b$, so the outlier decays faster and the overlap goes to 0. If $\eta\Lambda>1$, then $r_1=\eta\Lambda-1$, and $r_1>r_b$ iff $\eta(\Lambda+\mu)>2$. So the overlap tends to 1 exactly for
$$\frac{2}{\Lambda+\mu}<\eta<\frac{2}{\Lambda},$$
a narrow open window just below the stability limit. There the outlier component flips sign every step and shrinks only slowly. At $\eta=2/(\Lambda+\mu)$ exactly, $r_1=r_b$ and the overlap stays at its initial value.

**(d)** $r_1=|1-1.99|=0.99$ and $r_b=1-0.0199=0.9801$; note $\eta=0.0199>2/101\approx0.0198$, so we are in the window.

- $t=0$: overlap $=10^4/(10^4+999)\approx0.909$. One-step decrease: outlier $\tfrac12\cdot100\cdot(1-0.9801)=0.995$; bulk $\tfrac12\cdot999\cdot(1-0.9801^2)=499.5\times0.0394\approx19.68$. The bulk supplies $19.68/20.68\approx95\%$ of the decrease.
- $t=100$: $c_1^2=0.99^{200}\approx0.134$ and $c_i^2=0.9801^{200}\approx0.0180$ for the bulk. Overlap $=1340/(1340+17.9)\approx0.987$. Decrease: outlier $50\times0.134\times0.0199\approx0.133$; bulk $499.5\times0.0180\times0.0394\approx0.353$. The bulk still supplies about $73\%$.

So an overlap near 1 is compatible with most of the loss decrease coming from outside the top subspace: the gradient is dominated by a large, oscillating component that contributes little progress. (In this toy the outlier eventually dominates the decrease too, after about 150 steps, because it is the slowest-decaying direction.) The overlap measures where the gradient is, not where learning happens. The toy does not settle whether real training behaves this way; that takes intervention experiments on real networks.
"""
    ),

    # ------------------------------------------------------------------ 10
    markdown(
        head("10", "WB2 §1.D, WB2 §4.C.") + "\n\n" + reading(
            "An investigation into neural net optimization via Hessian eigenvalue density",
            "https://arxiv.org/abs/1901.10159",
            "Ghorbani, Krishnan & Xiao (ICML 2019)",
            "Estimates the full Hessian eigenvalue density of large networks, up to ImageNet scale, with stochastic Lanczos quadrature. "
            "In networks without batch normalization, large isolated eigenvalues appear early in training and the gradient concentrates in their eigenspace. "
            "With batch normalization these outliers are largely absent, which the authors connect to faster optimization.",
            importance=5, length=3,
        ) + r"""

**Exercise 10.1 (scale invariance).** This calculation is not in the paper. It shows one concrete way batch normalization changes curvature. When batch normalization uses the current batch's statistics, with the $\varepsilon$ in its denominator set to zero and nonzero pre-activation variance, rescaling a unit's nonzero incoming weight vector $w$ by $\alpha>0$ leaves its normalized output unchanged. (This fails with frozen inference statistics or an added weight penalty.) So, with all other parameters fixed, $L(\alpha w)=L(w)$ for every $\alpha>0$; assume $L$ is twice differentiable near $w$. Write $g=\nabla L(w)$ and $H=\nabla^2L(w)$, both taken with respect to $w$.

- (a) Show that $w^\top g=0$.
- (b) Show that $\nabla L(\alpha w)=\alpha^{-1}\nabla L(w)$ and $\nabla^2L(\alpha w)=\alpha^{-2}\nabla^2L(w)$.
- (c) Show that $Hw=-g$. Deduce that the curvature along $w$, $w^\top Hw$, is zero. When is $w$ an eigenvector of $H$ with eigenvalue 0, and why is it not one in general?
- (d) For a gradient step $w'=w-\eta g$, show that $\lVert w'\rVert^2=\lVert w\rVert^2+\eta^2\lVert g\rVert^2$. Using (b), separate the effect of changing $\lVert w\rVert$ from that of changing the unit direction $u=w/\lVert w\rVert$. What effective step size multiplies the gradient with respect to the direction?

Compare the layer rescaling in WB2 §1.D: as there, curvature measured in these weights is not a property of the function alone.
"""
    ),
    answer(
        r"""
**(a)** Differentiate $L(\alpha w)=L(w)$ with respect to $\alpha$: $w^\top\nabla L(\alpha w)=0$ for all $\alpha>0$. At $\alpha=1$, $w^\top g=0$. The gradient is always orthogonal to $w$.

**(b)** Differentiate $L(\alpha w)=L(w)$ with respect to $w$: $\alpha\nabla L(\alpha w)=\nabla L(w)$, so $\nabla L(\alpha w)=\alpha^{-1}\nabla L(w)$. Differentiate this identity with respect to $w$ again: $\alpha\,\nabla^2L(\alpha w)=\alpha^{-1}\nabla^2L(w)$, so $\nabla^2L(\alpha w)=\alpha^{-2}\nabla^2L(w)$.

**(c)** Differentiate $\nabla L(\alpha w)=\alpha^{-1}\nabla L(w)$ with respect to $\alpha$. The left side gives $\nabla^2L(\alpha w)\,w$ and the right side $-\alpha^{-2}\nabla L(w)$. At $\alpha=1$, $Hw=-g$. Then $w^\top Hw=-w^\top g=0$ by (a).

If $g=0$ (a stationary point), $Hw=0$, so $w$ is an eigenvector with eigenvalue 0. If $g\neq0$, $Hw=-g$ is nonzero and orthogonal to $w$, so it cannot be a multiple of $w$: $w$ is not an eigenvector. $H$ couples the radial direction to the gradient direction. Away from a minimum $H$ need not be positive semi-definite, so $w^\top Hw=0$ does not force $Hw=0$.

**(d)** $\lVert w-\eta g\rVert^2=\lVert w\rVert^2-2\eta\,w^\top g+\eta^2\lVert g\rVert^2=\lVert w\rVert^2+\eta^2\lVert g\rVert^2$, using (a). So under plain gradient descent (no weight decay) the weight norm never decreases. By (b) with $\alpha=\lVert w\rVert$ applied to $u=w/\lVert w\rVert$, $\nabla L(w)=\lVert w\rVert^{-1}\nabla L(u)$ and, exactly, $H(w)=\lVert w\rVert^{-2}H(u)$. To first order the unit direction moves as $u'\approx u-(\eta/\lVert w\rVert^2)\nabla L(u)$, so the effective step size on the function is $\eta/\lVert w\rVert^2$. Norm growth lowers this factor and the radial factor in $H(w_t)=\lVert w_t\rVert^{-2}H(u_t)$. It does not guarantee that the Hessian eigenvalues decrease during training, because $H(u_t)$ also changes as the direction moves. Weight decay works against the norm growth.
"""
    ),

    # ------------------------------------------------------------------ 11
    markdown(
        head("11", "WB1 §2.E (Exercise B.5, the KL-regularized optimum).") + "\n\n" + reading(
            "Reinforcement learning finetunes small subnetworks in large language models",
            "https://arxiv.org/abs/2505.11711",
            "Mukherjee, Yuan, Hakkani-Tür & Peng (NeurIPS 2025)",
            "Across the 7 RL algorithms and 10 LLMs tested, RL fine-tuning changes only 5–30% of the parameters by the paper's measure, and fine-tuning that subnetwork alone gives a nearly identical model. "
            "The subnetworks overlap across seeds, data and algorithms more than chance would predict, and the per-matrix updates are nearly full-rank. "
            "The authors conjecture that training on data near the policy's own distribution, rather than the KL penalty or gradient clipping, mainly causes the sparsity.",
            importance=5, length=3,
        ) + "\n\n" + reading(
            "RL's Razor: why online reinforcement learning forgets less",
            "https://arxiv.org/abs/2509.04259",
            "Shenfeld, Pari & Agrawal (2025)",
            "In the tested LLM and robotics settings, on-policy RL fine-tuning forgets less of prior capabilities than SFT at matched performance on the new task. "
            "The amount of forgetting is predicted by the KL divergence between the fine-tuned and base policies on the new task, and the authors argue that on-policy RL is implicitly biased toward small-KL solutions. "
            "In their setting the weight-update sparsity came from bfloat16 training and vanished in float32 at the same performance, with full-rank updates throughout (§6).",
            importance=4, length=3,
        ) + "\n\n" + reading(
            "On predictability of reinforcement learning dynamics for large language models",
            "https://arxiv.org/abs/2510.00553",
            "Cai et al. (2025)",
            "The low-rank side: across the 8 models and 7 algorithms tested, the top singular direction of each RL update matrix recovers over 99% of the reasoning gains (\"rank-1 dominance\"). "
            "That direction evolves roughly linearly over training, so the authors extrapolate the final update from an early window to speed training up.",
            importance=3, length=3,
        ) + r"""

**Exercise 11.1 (sparse is not low-rank).** An update $\Delta W\in\mathbb R^{n\times n}$ is *sparse* if most entries are zero, and has *rank at most $r$* if it can be written $\Delta W=AB^\top$ with $A,B\in\mathbb R^{n\times r}$. LoRA parametrizes the update in the second form with small $r$.

- (a) Give a matrix with only $n$ nonzero entries and rank $n$, and a matrix of rank 1 with all $n^2$ entries nonzero.
- (b) Fix a support $S\subseteq\{1,\dots,n\}^2$ that contains the positions $\{(i,\pi(i))\}$ of some permutation $\pi$. Show that for almost every choice of real values on $S$ (zero elsewhere), the matrix is invertible. *Hint:* $\det$ is a polynomial in the entries on $S$; find one choice of values where it is nonzero.
- (c) What does (b) suggest about how surprising "sparse but nearly full-rank" is? *Hint:* you may use the fact that a random $n\times n$ support in which each position is included independently with probability $\rho$ contains some permutation pattern with high probability once $\rho n\gg\log n$.
"""
    ),
    answer(
        r"""
**(a)** The identity $I_n$, or any permutation matrix with nonzero scalings, has $n$ nonzero entries and rank $n$. For $u,v\in\mathbb R^n$ with no zero entries, $uv^\top$ has entries $u_iv_j\neq0$ and rank 1.

**(b)** $\det M=\sum_\sigma\operatorname{sgn}(\sigma)\prod_iM_{i\sigma(i)}$ is a polynomial in the entries on $S$. Put 1 on the positions $(i,\pi(i))$ and 0 on the rest of $S$: the matrix is the permutation matrix of $\pi$, with determinant $\operatorname{sgn}(\pi)=\pm1$. So the polynomial is not identically zero. The zero set of a nonzero polynomial has Lebesgue measure zero, so $\det M\neq0$ for almost every choice of values on $S$.

**(c)** Sparsity alone does not imply low rank. By (b), generic values on a support containing a permutation pattern give a full-rank matrix almost surely. By the hint, a support with independent density $\rho=0.05$ contains such a pattern with high probability at large widths ($n=4096$ gives $\rho n\approx200$, against $\log n\approx8$). So "sparse but nearly full-rank" is what a generic sparse update looks like, not a coincidence. Learned supports and values can be structured or correlated, so the measured rank is still a separate empirical fact. If the sparsity survives a higher-precision measurement (Exercise 11.2), the informative findings are the small fraction of parameters that change and the overlap of that fraction across runs. A rank-$r$ update (LoRA's form) is a different restriction: it is generically dense.
"""
    ),
    markdown(
        r"""
**Exercise 11.2 (what "unchanged" means).** Mukherjee et al. count a parameter as unchanged when its bfloat16 values before and after fine-tuning differ by at most $10^{-5}$. A bfloat16 number has 8 exponent bits and 7 stored fraction bits, so 8 significant bits including the implicit leading 1.

- (a) Show that for normal (not subnormal) values with $2^e\le|w|<2^{e+1}$, the spacing between neighbouring bfloat16 numbers in that range is $2^{e-7}$.
- (b) For which magnitudes $|w|$ can a change of the stored value be within the $10^{-5}$ tolerance?
- (c) Suppose training keeps a higher-precision copy that starts exactly at the bfloat16 value $w$, with $|w|>2^{-9}$ and not a power of two, and the final weight $w+\Delta$ is rounded to the nearest bfloat16 number. Roughly how large must $|\Delta|$ be, relative to $|w|$, for the parameter to count as updated? Evaluate this for $|w|\approx0.02$.
- (d) The paper reports that SFT, measured the same way, leaves only 6–15% of parameters unchanged, against roughly 70–95% for RL. What does this comparison rule out, and what does it not?
"""
    ),
    answer(
        r"""
**(a)** The bfloat16 numbers in $[2^e,2^{e+1})$ are $2^e(1+f/2^7)$ for $f=0,\dots,127$, so neighbours differ by $2^e/2^7=2^{e-7}$.

**(b)** Away from powers of two, a change of the stored value is at least one spacing, $2^{e-7}$. This is at most $10^{-5}$ only if $e-7\le\log_2 10^{-5}\approx-16.6$, i.e. $e\le-10$, so $|w|<2^{-9}\approx0.002$. At $|w|=2^{-9}$ exactly, the neighbour toward zero is only $2^{-17}\approx7.6\times10^{-6}$ away (within tolerance), while the neighbour away from zero is $2^{-16}\approx1.5\times10^{-5}$ away (outside it). So some distinct stored value lies within tolerance iff $|w|\le2^{-9}$. For $|w|>2^{-9}$ the tolerance makes no difference: "unchanged" means the stored bfloat16 value is identical.

**(c)** Rounding to nearest returns $w$ when $|\Delta|$ is below half a spacing, $2^{e-8}$ (ignoring ties). Since $2^e\le|w|<2^{e+1}$, this threshold lies between $|w|/512$ and $|w|/256$: roughly 0.2–0.4% of $|w|$. For $|w|\approx0.02$ (bfloat16 stores $0.02$ as $0.02001953125$), $e=-6$ since $2^{-6}\approx0.0156$, so the threshold is $2^{-14}\approx6.1\times10^{-5}$, about 0.3% of $|w|$. So in this example a net movement below about 0.3% of $|w|$ leaves the stored value unchanged: "unchanged" does not mean "received no gradient" or "never moved". If updates were instead written straight into bfloat16 weights at every step, the same threshold would apply to each step separately, and many small steps could all be lost.

**(d)** It rules out the simplest artifact, that this measurement makes every fine-tune look sparse: under the same threshold, SFT looks dense. It does not distinguish "RL moves few parameters" from "RL moves most parameters by less than their rounding or absolute-tolerance threshold", because SFT may simply move weights further. Separating the two needs a higher-precision measurement. Shenfeld et al. (RL's Razor, §6) did this in their own setting: training in float32 gave the same performance and no sparsity, so there the sparsity was a bfloat16 effect. Their setting is not Mukherjee et al.'s, so this does not settle the question for every model, but it shows the concern is real. Mukherjee et al.'s finding that fine-tuning only the subnetwork gives a nearly identical model bears on whether any sub-threshold movement matters for the function.
"""
    ),

    # ------------------------------------------------------------------ 12
    markdown(
        head("12", "WB4 §4.") + "\n\n" + reading(
            "Physics of language models: Part 1, learning hierarchical language structures",
            "https://arxiv.org/abs/2305.13673",
            "Allen-Zhu & Li (2023; [project page](https://physics.allen-zhu.com/))",
            "Trains GPT-2-sized transformers on strings from deep, ambiguous synthetic context-free grammars, so the structure to be learned is known exactly. "
            "The models generate grammatical strings with high accuracy, probes recover the grammar's parse structure from hidden states, and attention patterns resemble what a dynamic-programming parser needs. "
            "The project page links the later parts of the series.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 13
    markdown(
        head("13", "WB5 §3 (shared representations).") + r"""

Two findings suggest that language models share more structure than their architectures force. A model with hidden size $d$ has next-token log-probabilities $\phi(h)^\top\psi(y)-\log Z(h)$, so the matrix of $\log p(y\mid h)$ over histories and next tokens has rank at most $d+1$ (the softmax bottleneck). For multi-token continuations nothing in the architecture bounds the rank, because the hidden state sees the earlier part of the continuation; Golowich et al. find that it is approximately low anyway, with linear relations among prompts that agree across models.

Subliminal learning is the stranger finding: a student fine-tuned on a teacher's innocuous outputs picks up the teacher's traits. Cloud et al.'s argument for why a shared initialization matters is short. If the teacher is $\theta_0+\varepsilon\Delta$ and the student starts at $\theta_0$, one small distillation step on *any* inputs moves the student by $\eta\varepsilon\,\mathbb E[J^\top HJ]\Delta+O(\varepsilon^2)$, with $J$ the output Jacobian at $\theta_0$ and $H\succeq0$ the loss Hessian in output space. ($H\succeq0$ because the loss is minimized when the student's outputs equal the teacher's.) To leading order in $\varepsilon$ that has a nonnegative inner product with $\Delta$, so the student moves toward the teacher whatever the data are about. With a different initialization the leading term instead fits the teacher's base behaviour, and the cross term $J_S^\top J_T$ has no sign. The log-linear account below gives a different mechanism that can work across architectures.

""" + reading(
            "Sequences of logits reveal the low rank structure of language models",
            "https://arxiv.org/abs/2510.24966",
            "Golowich, Liu & Shetty (2025)",
            "Matrices of log-probabilities of multi-token continuations given prompts are approximately low-rank across many modern language models, and the linear relations among prompts are consistent across models. "
            "The authors use this to generate a response to a target prompt from the model's outputs on unrelated or nonsensical prompts.",
            importance=4, length=3,
        ) + "\n\n" + reading(
            "Subliminal learning: language models transmit behavioral traits via hidden signals in data",
            "https://arxiv.org/abs/2507.14805",
            "Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks & Evans (2025)",
            "A student fine-tuned on number sequences, code or chain-of-thought generated by a teacher with a trait (a preferred animal, or misalignment) acquires the trait even when references to it are filtered out. "
            "The effect is reported only when teacher and student share a base model or initialization; an MNIST experiment illustrates the shared-initialization argument above. "
            "Short summary: [Subliminal learning (Anthropic Alignment Science blog)](https://alignment.anthropic.com/2025/subliminal-learning/).",
            importance=5, length=3,
        ) + "\n\n" + reading(
            "Subliminal learning is a LoRA artifact",
            "https://arxiv.org/abs/2606.00831",
            "Nief, Fu, Muchane & Holtzman (2026)",
            "Reports that subliminal learning in the number-sequence setting depends non-monotonically on LoRA rank, disappears under full fine-tuning, and is tied to tokens seen in both training and evaluation, such as the system prompt and chat template. "
            "Its arXiv listing states that follow-up work showed some of its main claims to be incorrect and that a substantial revision is in preparation, so it does not settle the question either way. Cloud et al.'s MNIST experiment uses no LoRA.",
            importance=2, length=2,
        ) + "\n\n" + reading(
            "Subliminal effects in your data: a general mechanism via log-linearity",
            "https://arxiv.org/abs/2602.04863",
            "Aden-Ali, Golowich, Liu, Shetty, Moitra & Haghtalab (2026)",
            "Assuming log-probabilities are approximately bilinear in a system-prompt embedding and a prompt–response embedding shared across models, the paper's Logit-Linear Selection picks subsets of an ordinary preference dataset that make a DPO-trained student behave as if given a target system prompt (preferring an animal, answering in a language absent from the data, adopting a persona). "
            "Each example's selection score is its projection onto the teacher's trait direction, so no single example states the trait. "
            "It reports the effect within a model and, more weakly, across architectures, in contrast with Cloud et al.",
            importance=4, length=3,
        )
    ),


]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb6", "06_mysteries_myriad"))
