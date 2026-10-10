#!/usr/bin/env python3
"""Build WB6, "Mysteries of Deep Learning": an annotated catalogue of strange empirical facts about trained networks.

Trail format (reviews/2026-10-09-trail-restructure-brief.md). No keystone: the
whole workbook is the catalogue. An intro (two sentences) and a one-line route,
then 13 sections with stable anchors `wb6-section-1` ... `wb6-section-13`
(TD notebooks and WB5 link to -4, -5, -9, -10, -11, -13). Each section opens
with one sentence stating the mystery, then one to three `reading()` blurbs of
about one sentence, sometimes a trailing "Also:" line. Four sections carry a
"### Exercises: <topic>" subsection with self-contained exercises and folded
solutions: §5 (packing almost-orthogonal directions; the two-feature toy behind
the interactive figure), §9 (quadratic toy for gradient overlap), §10 (scale
invariance under batch norm), §11 (sparse vs low-rank updates; what "unchanged"
means in bfloat16). Pre-restructure builder: _pre_trail_2026-10-09/build_wb6.py.
Earlier editorial record: reviews/wb6-editorial.md.
"""
from nbkit import answer, figure, markdown, reading, run

SP = "assets/interactive/superposition.html"

SECTIONS = [
    ("1", "Transformer middle layers", "the best features are not at the top."),
    ("2", "Bottom-up learning", "lower layers settle first."),
    ("3", "Model stitching works", "one linear layer joins two different networks."),
    ("4", "Critical periods", "an early deficit outlasts later training."),
    ("5", "Superposition and adversarial attacks (and on you too)", "more features than dimensions; brittle but real features. Exercises, interactive figure."),
    ("6", "Critical modules", "most layers can be reset to initialization; a few cannot."),
    ("7", "Distillation", "soft outputs teach far more than labels."),
    ("8", "Rewinding and lottery tickets", "sparse subnetworks that train alone, if the initialization is kept."),
    ("9", "Gradient descent in a tiny subspace", "the gradient lives in a few Hessian directions. Exercise."),
    ("10", "Spectra and curvature", "Hessian bulk plus outliers; batch norm removes the outliers. Exercise."),
    ("11", "Low-rank RL updates and the KL leash", "updates sparse yet full-rank, or rank-1 dominated; RL forgets less than SFT. Exercises."),
    ("12", "Physics of language models", "transformers learn to parse a synthetic grammar."),
    ("13", "Hidden signals: low-rank logits and subliminal learning", "shared low-rank structure; traits passed through unrelated data."),
]

TITLES = {k: t for k, t, _ in SECTIONS}
ROUTE = "\n".join(f"- [§{k}. {t}](#wb6-section-{k}): {p}" for k, t, p in SECTIONS)


def head(k: str, mystery: str) -> str:
    return f'<a id="wb6-section-{k}"></a>\n\n## {k}. {TITLES[k]}\n\n{mystery}'


CELLS = [
    markdown(
        r"""
# Workbook 6 — Mysteries of Deep Learning

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Trained networks do many things that no theory predicted, and any account of deep learning has to explain them. This workbook is a catalogue of such facts, one per section with the papers that report them: read until you find something you didn't know.

""" + ROUTE + r"""

Sections are independent. Solutions to the exercises are folded under "Solution". The Training Dynamics notebooks take up §4 and §8–§11.
"""
    ),

    # ------------------------------------------------------------------ 1
    markdown(
        head("1", "Intermediate layers of language models often give better general-purpose features than the final layer, even though the final layer is the one the training objective shapes most directly.")
        + reading(
            "Layer by layer: uncovering hidden representations in language models",
            "https://arxiv.org/abs/2502.02013",
            "Skean, Arefin, Zhao, Patel, Naghiyev, LeCun & Shwartz-Ziv (ICML 2025)",
            "On 32 embedding tasks, intermediate layers of transformers and state-space models often beat the last layer; label-free measures of representation quality track how this varies with depth.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 2
    markdown(
        head("2", "Gradient descent updates every layer at once, yet lower layers converge first and upper layers later: networks learn bottom-up.")
        + reading(
            "SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability",
            "https://arxiv.org/abs/1706.05806",
            "Raghu, Gilmer, Yosinski & Sohl-Dickstein (NeurIPS 2017)",
            "Introduces SVCCA for comparing layer representations and finds low effective dimensionality and bottom-up convergence, which motivates freezing lower layers in sequence to save compute.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 3
    markdown(
        head("3", "The bottom of one trained network joins the top of another through a single linear layer with little loss, so separately trained networks represent things almost the same way (see WB5, representation similarity).")
        + reading(
            "Revisiting model stitching to compare neural representations",
            "https://arxiv.org/abs/2106.07682",
            "Bansal, Nakkiran & Barak (NeurIPS 2021)",
            "Networks from different seeds, or supervised versus self-supervised training, stitch with little loss; representations learned with more data, width or training improve weaker models when stitched in.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 4
    markdown(
        head("4", "As in animal vision, a deficit early in training (blurred images) leaves a permanent deficit, while the same deficit later does little harm: later training does not undo what was decided early.")
        + reading(
            "Critical learning periods in deep neural networks",
            "https://arxiv.org/abs/1711.08856",
            "Achille, Rovere & Soatto (ICLR 2019)",
            "Early blur leaves a lasting deficit but early vertical flips do not; the Fisher information in the weights rises then falls during training, which the authors tie to the critical period.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 5
    markdown(
        head("5", "Networks store more sparse features than they have dimensions, in almost-orthogonal directions; adversarial examples may exploit this interference, or features that are predictive but brittle, and some even bias briefly shown humans (see WB5, linear representation hypothesis).")
        + reading(
            "Toy models of superposition",
            "https://transformer-circuits.pub/2022/toy_model/index.html",
            "Elhage et al. (Anthropic, 2022; also arXiv 2209.10652)",
            "Toy ReLU autoencoders store sparse features in superposition, with a phase change in sparsity and importance and features arranged as regular polytopes; 'Superposition as a Phase Change' has the figure's phase diagram.",
            importance=5, length=4,
        ) + reading(
            "Adversarial examples are not bugs, they are features",
            "https://arxiv.org/abs/1905.02175",
            "Ilyas, Santurkar, Tsipras, Engstrom, Tran & Madry (NeurIPS 2019)",
            "Training only on adversarial images relabelled with the attack's target class, which look mislabelled to humans, still gives non-trivial clean accuracy: the perturbations carry real, predictive, non-robust features.",
            importance=4, length=3,
        ) + reading(
            "Adversarial examples that fool both computer vision and time-limited humans",
            "https://arxiv.org/abs/1802.08195",
            "Elsayed, Shankar, Cheung, Papernot, Kurakin, Goodfellow & Sohl-Dickstein (NeurIPS 2018)",
            "Perturbations built to transfer across an ensemble of models with a retina-like front end measurably bias people's choices when images are shown for a fraction of a second.",
            importance=2, length=2,
        ) + "\n\n" + figure(SP, r"Panel 1 is the two-feature, one-dimension phase diagram over density and relative importance (dashed: the closed form of Exercise 5.2; solid: optimized weights and biases; dots: trained models). Panel 2 trains $n=5$ or $6$ features in $m=2$ dimensions in your browser: as sparsity grows, features go from dedicated dimensions to antipodal pairs to a regular polygon.")
        + r"""

### Exercises: packing directions and the two-feature toy

**Exercise 5.1 (packing almost-orthogonal directions).** Let $u,v$ be independent, uniformly random unit vectors in $\mathbb R^m$.

- (a) Show that $\mathbb E\langle u,v\rangle=0$ and $\mathbb E\langle u,v\rangle^2=1/m$.
- (b) Take as given $\Pr(|\langle u,v\rangle|>\varepsilon)\le2e^{-m\varepsilon^2/2}$. Show that if $N^2e^{-m\varepsilon^2/2}<1$, then $N$ independent random unit vectors have all pairwise $|\cos|\le\varepsilon$ with positive probability. Conclude that $\mathbb R^m$ contains about $e^{m\varepsilon^2/4}$ such directions.

<details><summary>Hint</summary>

(a) Condition on $v$ and use $\mathbb E[uu^\top]=I/m$ (isotropy, and the trace is 1). (b) Union bound over pairs.

</details>
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
**Exercise 5.2 (the smallest superposition).** The toy model of Elhage et al. compresses features $x\in\mathbb R^n$ to $Wx\in\mathbb R^m$ and reads them back as $\hat x=\mathrm{ReLU}(W^\top Wx+b)$, with loss $\sum_iI_i\,\mathbb E(\hat x_i-x_i)^2$ for importances $I_i$. Take $n=2$, $m=1$, importances $1$ and $r\le1$, and features that are independently $0$ with probability $1-p$ and uniform on $[0,1]$ with probability $p$.

- (a) *Dedicated:* $W=[1,0]$, $b=(0,b_2)$ with the best $b_2\ge0$. Show that the loss is $r\,(p/3-p^2/4)$.
- (b) *Antipodal:* $W=[1,-1]$, $b=0$. Show that the loss is $(1+r)\,p^2/6$.
- (c) Show that antipodal storage wins iff $p<4r/(2+5r)$. What does this say about sparse features, and about dense ones?
"""
    ),
    answer(
        r"""
**(a)** $\hat x_1=\mathrm{ReLU}(x_1)=x_1$, so feature 1 costs nothing. $\hat x_2=\mathrm{ReLU}(b_2)=b_2$ is a constant; the best constant is $\mathbb Ex_2=p/2$, leaving $r\operatorname{Var}x_2=r(\mathbb Ex_2^2-(\mathbb Ex_2)^2)=r(p/3-p^2/4)$.

**(b)** $W^\top W=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$, so $\hat x_1=\mathrm{ReLU}(x_1-x_2)$ and $\hat x_2=\mathrm{ReLU}(x_2-x_1)$. If at most one feature is active, reconstruction is exact: the ReLU erases the negative interference. If both are active and $x_1>x_2$, then $\hat x_1=x_1-x_2$ and $\hat x_2=0$, an error of $(1+r)x_2^2$; symmetrically otherwise. So the loss is $p^2(1+r)\,\mathbb E[\min(x_1,x_2)^2]$. The minimum of two uniforms has density $2(1-s)$ on $[0,1]$, so $\mathbb E\min^2=\int_0^12s^2(1-s)\,ds=1/6$.

**(c)** $(1+r)p^2/6<r(p/3-p^2/4)$. Divide by $p/12$: $2(1+r)p<4r-3rp$, i.e. $p(2+5r)<4r$. Sparse features ($p\to0$) are superposed for any $r>0$, even when one matters little, because interference costs $O(p^2)$ against $O(p)$ for dropping a feature. Dense features force a choice. At $r=1$ the threshold is $p=4/7$. This is the dashed boundary in panel 1 of the figure.
"""
    ),

    # ------------------------------------------------------------------ 6
    markdown(
        head("6", "After training, most layers can be reset to their initial weights with almost no loss, but a few cannot: the trained function depends on its weights very unevenly.")
        + reading(
            "Are all layers created equal?",
            "https://arxiv.org/abs/1902.01996",
            "Zhang, Bengio & Singer (2019; JMLR 2022)",
            "Resets one layer at a time to its initial or to fresh random weights: most layers are \"robust\" to re-initialization, a few are \"critical\", and which ones depends on the architecture.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 7
    markdown(
        head("7", "A small network trained on a large network's softened outputs learns far more than from the labels, even about a class it never sees, so the probabilities assigned to wrong answers carry much of what the teacher knows.")
        + reading(
            "Distilling the knowledge in a neural network",
            "https://arxiv.org/abs/1503.02531",
            "Hinton, Vinyals & Dean (2015)",
            "Temperature-softened teacher outputs transfer most of a large model's or ensemble's performance; a student whose transfer set omits every 3 still classifies most test 3s correctly.",
            importance=5, length=2,
        )
    ),

    # ------------------------------------------------------------------ 8
    markdown(
        head("8", "Trained networks contain sparse subnetworks that, rewound to their initial weights, train alone to full accuracy but do worse with fresh random weights: the particular initialization matters.")
        + reading(
            "The lottery ticket hypothesis: finding sparse, trainable neural networks",
            "https://arxiv.org/abs/1803.03635",
            "Frankle & Carbin (ICLR 2019)",
            "Iterative magnitude pruning with rewinding to initialization finds subnetworks under 10–20% of the original size that match full accuracy on MNIST and CIFAR-10; re-randomized, they do worse.",
            importance=5, length=3,
        ) + reading(
            "Linear mode connectivity and the lottery ticket hypothesis",
            "https://arxiv.org/abs/1912.05671",
            "Frankle, Dziugaite, Roy & Carbin (ICML 2020)",
            "Tickets work only from the point where training becomes stable to SGD noise (two runs with different data orders end up linearly connected): at initialization for small networks, early in training for ResNet-50.",
            importance=4, length=3,
        )
    ),

    # ------------------------------------------------------------------ 9
    markdown(
        head("9", "After a short initial phase the gradient lies almost entirely in the span of a few top Hessian eigenvectors, about as many as there are classes, in networks with millions of parameters.")
        + reading(
            "Gradient descent happens in a tiny subspace",
            "https://arxiv.org/abs/1812.04754",
            "Gur-Ari, Roberts & Dyer (2018)",
            "Measures the gradient's overlap with the top Hessian eigenspace during training; the subspace is small, changes slowly, and a short toy model explains the count.",
            importance=5, length=2,
        ) + r"""

### Exercises: where the gradient is versus where learning happens

**Exercise 9.1 (a quadratic toy).** This toy is not the paper's model. It shows that "the gradient lies in the top subspace" and "the loss goes down along the top subspace" can come apart. Take $L(\theta)=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$ with $H$ symmetric positive semi-definite, orthonormal eigenvectors $v_i$ with eigenvalues $\lambda_1\ge\lambda_2\ge\dots$, and write $\theta-\theta^*=\sum_i c_iv_i$. The overlap is $\lVert P_kg\rVert^2/\lVert g\rVert^2$, where $g=\nabla L$ and $P_k$ projects onto $v_1,\dots,v_k$.

- (a) Show that the overlap equals $\sum_{i\le k}\lambda_i^2c_i^2\big/\sum_i\lambda_i^2c_i^2$, and that $L=\tfrac12\sum_i\lambda_ic_i^2$.
- (b) For gradient descent $\theta\leftarrow\theta-\eta g$, show that $c_i(t)=(1-\eta\lambda_i)^t\,c_i(0)$.
- (c) Take one outlier $\lambda_1=\Lambda$ and $p-1$ bulk directions with $\lambda_i=\mu$, where $0<\mu\le\Lambda/2$, and $k=1$. Assume $c_1(0)\neq0$ and some bulk $c_i(0)\neq0$. For which step sizes $0<\eta<2/\Lambda$ does the overlap tend to 1 as $t\to\infty$?
- (d) Take $\Lambda=100$, $\mu=1$, $p=1000$, $\eta=0.0199$ and all $c_i(0)=1$. At $t=0$ and at $t=100$, compute the overlap and the fraction of the one-step loss decrease that comes from the bulk.

<details><summary>Hint</summary>

For (d): direction $i$ contributes $\tfrac12\lambda_ic_i^2\bigl(1-(1-\eta\lambda_i)^2\bigr)$ to the decrease in one step.

</details>
"""
    ),
    answer(
        r"""
**(a)** $g=H(\theta-\theta^*)=\sum_i\lambda_ic_iv_i$ and $P_kg=\sum_{i\le k}\lambda_ic_iv_i$. By orthonormality, $\lVert g\rVert^2=\sum_i\lambda_i^2c_i^2$ and $\lVert P_kg\rVert^2=\sum_{i\le k}\lambda_i^2c_i^2$. Likewise $L=\tfrac12\sum_i\lambda_ic_i^2$.

**(b)** $\theta-\theta^*\leftarrow(I-\eta H)(\theta-\theta^*)$, and $v_i$ is an eigenvector of $I-\eta H$ with eigenvalue $1-\eta\lambda_i$. So $c_i\leftarrow(1-\eta\lambda_i)c_i$ at each step.

**(c)** Let $r_1=|1-\eta\Lambda|$ and $r_b=1-\eta\mu$. Since $\eta<2/\Lambda$ and $\mu\le\Lambda/2$, $0<r_b<1$. By (a) and (b), with $k=1$,
$$\text{overlap}(t)=\frac{\Lambda^2c_1(0)^2\,r_1^{2t}}{\Lambda^2c_1(0)^2\,r_1^{2t}+\mu^2\sum_{i\ge2}c_i(0)^2\,r_b^{2t}},$$
which tends to 1 if $r_1>r_b$ and to 0 if $r_1<r_b$. If $\eta\Lambda\le1$, then $r_1=1-\eta\Lambda<r_b$, so the overlap goes to 0. If $\eta\Lambda>1$, then $r_1=\eta\Lambda-1$, and $r_1>r_b$ iff $\eta(\Lambda+\mu)>2$. So the overlap tends to 1 exactly for
$$\frac{2}{\Lambda+\mu}<\eta<\frac{2}{\Lambda},$$
a narrow window just below the stability limit, where the outlier component flips sign every step and shrinks slowly. At $\eta=2/(\Lambda+\mu)$ exactly, $r_1=r_b$ and the overlap stays at its initial value.

**(d)** $r_1=0.99$ and $r_b=0.9801$; $\eta=0.0199>2/101\approx0.0198$, so we are in the window.

- $t=0$: overlap $=10^4/(10^4+999)\approx0.909$. One-step decrease: outlier $\tfrac12\cdot100\cdot(1-0.9801)=0.995$; bulk $\tfrac12\cdot999\cdot(1-0.9801^2)\approx19.68$. The bulk supplies about $95\%$.
- $t=100$: $c_1^2=0.99^{200}\approx0.134$ and bulk $c_i^2=0.9801^{200}\approx0.0180$. Overlap $=1340/(1340+17.9)\approx0.987$. Decrease: outlier $\approx0.133$, bulk $\approx0.353$. The bulk still supplies about $73\%$.

So an overlap near 1 is compatible with most of the loss decrease coming from outside the top subspace: the gradient is dominated by a large oscillating component that contributes little progress. (Here the outlier eventually dominates the decrease too, after about 150 steps, since it decays slowest.) The overlap measures where the gradient is, not where learning happens; whether real training behaves like this takes intervention experiments on real networks.
"""
    ),

    # ------------------------------------------------------------------ 10
    markdown(
        head("10", "The Hessian spectrum of a trained network is a bulk near zero plus a few large outliers that appear early in training, and batch normalization largely removes the outliers.")
        + reading(
            "An investigation into neural net optimization via Hessian eigenvalue density",
            "https://arxiv.org/abs/1901.10159",
            "Ghorbani, Krishnan & Xiao (ICML 2019)",
            "Estimates full Hessian spectra up to ImageNet scale with stochastic Lanczos; without batch norm, early outliers capture the gradient, and their absence with batch norm is linked to faster optimization.",
            importance=5, length=3,
        ) + r"""

### Exercises: scale invariance under batch norm

**Exercise 10.1 (scale invariance).** This calculation is not in the paper; it shows one concrete way batch normalization changes curvature. With batch statistics (and $\varepsilon=0$, nonzero pre-activation variance), rescaling a unit's nonzero incoming weight vector $w$ by $\alpha>0$ leaves its normalized output unchanged. So, with all other parameters fixed, $L(\alpha w)=L(w)$ for every $\alpha>0$; assume $L$ is twice differentiable near $w$, and write $g=\nabla L(w)$, $H=\nabla^2L(w)$ with respect to $w$.

- (a) Show that $w^\top g=0$.
- (b) Show that $\nabla L(\alpha w)=\alpha^{-1}\nabla L(w)$ and $\nabla^2L(\alpha w)=\alpha^{-2}\nabla^2L(w)$.
- (c) Show that $Hw=-g$. Deduce that $w^\top Hw=0$. When is $w$ an eigenvector of $H$ with eigenvalue 0, and why is it not one in general?
- (d) For a gradient step $w'=w-\eta g$, show that $\lVert w'\rVert^2=\lVert w\rVert^2+\eta^2\lVert g\rVert^2$. Using (b), separate the effect of changing $\lVert w\rVert$ from that of changing the direction $u=w/\lVert w\rVert$. What effective step size acts on the direction?

As with layer rescaling in ReLU networks (Dinh et al., *Sharp minima can generalize*, in WB2), curvature measured in these weights is not a property of the function alone.
"""
    ),
    answer(
        r"""
**(a)** Differentiate $L(\alpha w)=L(w)$ with respect to $\alpha$: $w^\top\nabla L(\alpha w)=0$ for all $\alpha>0$. At $\alpha=1$, $w^\top g=0$.

**(b)** Differentiate $L(\alpha w)=L(w)$ with respect to $w$: $\alpha\nabla L(\alpha w)=\nabla L(w)$. Differentiate again: $\alpha^2\nabla^2L(\alpha w)=\nabla^2L(w)$.

**(c)** Differentiate $\nabla L(\alpha w)=\alpha^{-1}\nabla L(w)$ with respect to $\alpha$: $\nabla^2L(\alpha w)\,w=-\alpha^{-2}\nabla L(w)$. At $\alpha=1$, $Hw=-g$, so $w^\top Hw=-w^\top g=0$.

If $g=0$ (a stationary point), $Hw=0$ and $w$ is an eigenvector with eigenvalue 0. If $g\neq0$, $Hw=-g$ is nonzero and orthogonal to $w$, so $w$ is not an eigenvector: $H$ couples the radial direction to the gradient direction. Away from a minimum $H$ need not be positive semi-definite, so $w^\top Hw=0$ does not force $Hw=0$.

**(d)** $\lVert w-\eta g\rVert^2=\lVert w\rVert^2-2\eta\,w^\top g+\eta^2\lVert g\rVert^2=\lVert w\rVert^2+\eta^2\lVert g\rVert^2$ by (a): without weight decay the norm never decreases. By (b) with $\alpha=\lVert w\rVert$, $\nabla L(w)=\lVert w\rVert^{-1}\nabla L(u)$ and $H(w)=\lVert w\rVert^{-2}H(u)$. To first order $u'\approx u-(\eta/\lVert w\rVert^2)\nabla L(u)$, so the effective step size is $\eta/\lVert w\rVert^2$. Norm growth shrinks this factor and the radial factor in $H(w_t)=\lVert w_t\rVert^{-2}H(u_t)$; it does not guarantee the Hessian eigenvalues fall, because $H(u_t)$ changes too. Weight decay works against the norm growth.
"""
    ),

    # ------------------------------------------------------------------ 11
    markdown(
        head("11", "RL fine-tuning of LLMs is reported to change few parameters yet at full rank, or to be dominated by one rank-1 direction, and to forget less than SFT; how these fit together, and how much is a bfloat16 artifact, is open (KL-regularized optimum: WB1, planning as inference).")
        + reading(
            "Reinforcement learning finetunes small subnetworks in large language models",
            "https://arxiv.org/abs/2505.11711",
            "Mukherjee, Yuan, Hakkani-Tür & Peng (NeurIPS 2025)",
            "Across 7 RL algorithms and 10 LLMs, RL changes 5–30% of parameters, nearly full-rank per matrix, overlapping across seeds and algorithms; fine-tuning that subnetwork alone gives a nearly identical model.",
            importance=5, length=3,
        ) + reading(
            "RL's Razor: why online reinforcement learning forgets less",
            "https://arxiv.org/abs/2509.04259",
            "Shenfeld, Pari & Agrawal (2025)",
            "On-policy RL forgets less than SFT at matched new-task performance, with forgetting predicted by KL to the base policy; their §6 finds the update sparsity was a bfloat16 effect.",
            importance=4, length=3,
        ) + reading(
            "On predictability of reinforcement learning dynamics for large language models",
            "https://arxiv.org/abs/2510.00553",
            "Cai et al. (2025)",
            "The top singular direction of each RL update matrix recovers over 99% of the reasoning gains and evolves almost linearly, so the final update can be extrapolated from an early window.",
            importance=3, length=3,
        ) + r"""

### Exercises: sparse updates and bfloat16

**Exercise 11.1 (sparse is not low-rank).** An update $\Delta W\in\mathbb R^{n\times n}$ is *sparse* if most entries are zero, and has *rank at most $r$* if $\Delta W=AB^\top$ with $A,B\in\mathbb R^{n\times r}$ (LoRA's form, with small $r$).

- (a) Give a matrix with only $n$ nonzero entries and rank $n$, and a matrix of rank 1 with all $n^2$ entries nonzero.
- (b) Fix a support $S\subseteq\{1,\dots,n\}^2$ that contains the positions $\{(i,\pi(i))\}$ of some permutation $\pi$. Show that for almost every choice of real values on $S$ (zero elsewhere), the matrix is invertible.
- (c) What does (b) suggest about how surprising "sparse but nearly full-rank" is? You may use that a random support including each position independently with probability $\rho$ contains some permutation pattern with high probability once $\rho n\gg\log n$.

<details><summary>Hint</summary>

For (b): $\det$ is a polynomial in the entries on $S$; find one choice of values where it is nonzero.

</details>
"""
    ),
    answer(
        r"""
**(a)** The identity $I_n$ (or any scaled permutation matrix) has $n$ nonzero entries and rank $n$. For $u,v\in\mathbb R^n$ with no zero entries, $uv^\top$ has all entries $u_iv_j\neq0$ and rank 1.

**(b)** $\det M=\sum_\sigma\operatorname{sgn}(\sigma)\prod_iM_{i\sigma(i)}$ is a polynomial in the entries on $S$. Putting 1 on the positions $(i,\pi(i))$ and 0 elsewhere on $S$ gives the permutation matrix of $\pi$, with determinant $\pm1$, so the polynomial is not identically zero. The zero set of a nonzero polynomial has Lebesgue measure zero, so $\det M\neq0$ for almost every choice of values.

**(c)** Sparsity alone does not imply low rank: generic values on a support containing a permutation pattern give full rank almost surely, and a support with density $\rho=0.05$ at width $n=4096$ has $\rho n\approx200\gg\log n\approx8$. So "sparse but nearly full-rank" is what a generic sparse update looks like. Learned supports and values can be structured, so the measured rank is still an empirical fact, but the informative findings are the small fraction of parameters that change (if it survives a higher-precision measurement, Exercise 11.2) and its overlap across runs. A rank-$r$ update is a different restriction: it is generically dense.
"""
    ),
    markdown(
        r"""
**Exercise 11.2 (what "unchanged" means).** Mukherjee et al. count a parameter as unchanged when its bfloat16 values before and after fine-tuning differ by at most $10^{-5}$. A bfloat16 number has 8 exponent bits and 7 stored fraction bits (8 significant bits with the implicit leading 1).

- (a) Show that for normal values with $2^e\le|w|<2^{e+1}$, the spacing between neighbouring bfloat16 numbers is $2^{e-7}$.
- (b) For which magnitudes $|w|$ can a change of the stored value be within the $10^{-5}$ tolerance?
- (c) Suppose training keeps a higher-precision copy that starts exactly at the bfloat16 value $w$ ($|w|>2^{-9}$, not a power of two), and the final weight $w+\Delta$ is rounded to the nearest bfloat16 number. Roughly how large must $|\Delta|/|w|$ be for the parameter to count as updated? Evaluate for $|w|\approx0.02$.
- (d) Measured the same way, SFT leaves only 6–15% of parameters unchanged, against roughly 70–95% for RL. What does this comparison rule out, and what does it not?
"""
    ),
    answer(
        r"""
**(a)** The bfloat16 numbers in $[2^e,2^{e+1})$ are $2^e(1+f/2^7)$ for $f=0,\dots,127$, so neighbours differ by $2^{e-7}$.

**(b)** Away from powers of two, a change of stored value is at least one spacing, $2^{e-7}$, which is $\le10^{-5}$ only if $e\le-10$, i.e. $|w|<2^{-9}\approx0.002$. At $|w|=2^{-9}$ exactly, the neighbour toward zero is $2^{-17}\approx7.6\times10^{-6}$ away (within tolerance) and the one away from zero $2^{-16}\approx1.5\times10^{-5}$ (outside). So a distinct stored value lies within tolerance iff $|w|\le2^{-9}$. Above that, "unchanged" means the stored bfloat16 value is identical.

**(c)** Rounding returns $w$ when $|\Delta|$ is below half a spacing, $2^{e-8}$, which lies between $|w|/512$ and $|w|/256$: roughly 0.2–0.4% of $|w|$. For $|w|\approx0.02$, $e=-6$, so the threshold is $2^{-14}\approx6.1\times10^{-5}$, about 0.3%. "Unchanged" does not mean "received no gradient". If updates were written straight into bfloat16 weights every step, the same threshold would apply per step, and many small steps could all be lost.

**(d)** It rules out the simplest artifact, that this measurement makes every fine-tune look sparse: SFT looks dense under it. It does not distinguish "RL moves few parameters" from "RL moves most parameters by less than the rounding threshold", since SFT may simply move weights further. That needs a higher-precision measurement. Shenfeld et al. (RL's Razor, §6) did this in their setting: float32 training gave the same performance and no sparsity. Their setting is not Mukherjee et al.'s, so this does not settle it everywhere, but the concern is real. Mukherjee et al.'s finding that tuning only the subnetwork gives a nearly identical model bears on whether sub-threshold movement matters for the function.
"""
    ),

    # ------------------------------------------------------------------ 12
    markdown(
        head("12", "Transformers trained on a deep, ambiguous synthetic grammar learn to parse it, though nothing in the architecture is built for parsing (data-model side: WB4, hierarchical representations).")
        + reading(
            "Physics of language models: Part 1, learning hierarchical language structures",
            "https://arxiv.org/abs/2305.13673",
            "Allen-Zhu & Li (2023; [project page](https://physics.allen-zhu.com/), with the later parts)",
            "GPT-2-sized models trained on context-free grammars generate grammatical strings; linear probes recover the parse, and attention patterns match what a dynamic-programming parser needs.",
            importance=5, length=3,
        )
    ),

    # ------------------------------------------------------------------ 13
    markdown(
        head("13", "Language models' multi-token log-probabilities are approximately low-rank and share structure across models, though the architecture forces low rank only for the next token; and a student fine-tuned on a teacher's innocuous outputs picks up the teacher's traits.")
        + reading(
            "Sequences of logits reveal the low rank structure of language models",
            "https://arxiv.org/abs/2510.24966",
            "Golowich, Liu & Shetty (2025)",
            "Prompt-by-continuation log-probability matrices are approximately low-rank, with linear relations among prompts consistent across models; this lets one answer a target prompt from outputs on unrelated or nonsense prompts.",
            importance=4, length=3,
        ) + reading(
            "Subliminal learning: language models transmit behavioral traits via hidden signals in data",
            "https://arxiv.org/abs/2507.14805",
            "Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks & Evans (2025; [blog summary](https://alignment.anthropic.com/2025/subliminal-learning/))",
            "Traits (a favourite animal, misalignment) pass through filtered numbers, code or chain-of-thought, but only when teacher and student share an initialization; a short argument shows why, to first order.",
            importance=5, length=3,
        ) + reading(
            "Subliminal effects in your data: a general mechanism via log-linearity",
            "https://arxiv.org/abs/2602.04863",
            "Aden-Ali, Golowich, Liu, Shetty, Moitra & Haghtalab (2026)",
            "If log-probabilities are roughly bilinear in shared embeddings, selecting preference data by projection onto a trait direction instils the trait through DPO, within a model and more weakly across architectures.",
            importance=4, length=3,
        ) + r"""

Also: [Subliminal learning is a LoRA artifact](https://arxiv.org/abs/2606.00831) (Nief et al. 2026; its arXiv listing says follow-up work showed some main claims incorrect, and Cloud et al.'s MNIST experiment uses no LoRA).
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb6", "06_mysteries_myriad"))
