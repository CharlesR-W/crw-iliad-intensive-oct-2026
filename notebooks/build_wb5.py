#!/usr/bin/env python3
"""Build WB5 (Deep Learning and Universalities: Scaling Laws and Architectures).

Shape (trail restructure, 2026-10-09 afternoon; reviews/2026-10-09-trail-restructure-brief.md):
§1 is the keystone, Kaplan et al. 2020, with two short core ideas (smooth power laws;
architecture mostly sets constants, not exponents). §2-4 are lightly annotated reading
lists, each opening with a one-sentence purpose and ending in "### Exercises: ..." with
self-contained exercises and folded solutions: §2 why power laws (spectral/ridge and quanta
learning curves), §3 architecture doesn't matter much (Fermi on compute multipliers),
§4 representational convergence (CKA/CCA, PRH, LRH; invariance exercises).
Chinchilla and compute-optimal allocation live in WB3. Superposition and low-rank logits
live in WB6. Pre-trail snapshot: _pre_trail_2026-10-09/; older material, including the
full perfect-PRH derivation, in _old_wb5/. Standard library only.
"""
from nbkit import answer, figure, markdown, reading, run

SF = "assets/interactive/spectral_filters.html"

CELLS = [
    # ------------------------------------------------------------------ intro
    markdown(r"""
# Workbook 5 — Deep Learning and Universalities: Scaling Laws and Architectures

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Different networks trained on similar data end up with the same smooth loss curves, nearly regardless of architecture, and arguably with similar internal representations. Nobody designed these regularities in, and nobody fully knows why they hold: the machines, they just want to learn.

- [1. Keystone: scaling laws](#wb5-keystone). Kaplan et al. 2020.
- [2. Why power laws? A hundred unnecessary explanations](#wb5-section-2). Exercises on power-law learning curves.
- [3. Another bitter lesson: architecture doesn't matter (much)](#wb5-section-3). Fermi exercise on compute multipliers.
- [4. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses](#wb5-section-4). Exercises on similarity invariances.
"""),

    # ------------------------------------------------------------------ §1 keystone
    markdown(r"""
<a id="wb5-keystone"></a>

## 1. Keystone: scaling laws

Why it's here: the loss of a language model is a predictable, smooth function of scale, and the details of the network barely enter. Both facts are empirical, and both are unexplained.

""" + reading(
        "Scaling laws for neural language models",
        "https://arxiv.org/abs/2001.08361",
        "Kaplan et al. (2020)",
        "Test loss is a power law in parameters, data and compute, while width, depth and shape barely matter at fixed parameter count. "
        "The abstract, Section 3 and Figures 1 and 5–7 carry the argument.",
        importance=5,
        length=3,
    ) + r"""

**Smooth power laws nobody designed in.** Kaplan et al. train transformer language models with up to about $10^9$ non-embedding parameters $N$. When not bottlenecked by the others, test loss falls as $L(N)\approx(N_c/N)^{\alpha_N}$ with $\alpha_N\approx0.076$, and as similar power laws in dataset size $D$ ($\alpha_D\approx0.095$) and in compute ($\alpha_C\approx0.050$), with some trends spanning more than seven orders of magnitude. Loss on other text datasets follows the same power with an offset. Nothing in the architecture, the optimizer or the data was chosen to produce this. It made training a much larger model a predictable investment rather than a gamble. Fitting these curves and allocating compute between parameters and data (Chinchilla) is in WB3, scale section.

**Architecture mostly sets constants.** At fixed $N$, the loss varies only a few percent over a wide range of shapes. The aspect ratio $d_{\rm model}/n_{\rm layer}$ can vary by a factor of 40 at little cost (Figure 5), and models of different depths fall on one curve once embedding parameters are excluded (Figure 6). [§3](#wb5-section-3) extends this across architecture families: most changes shift the curve by a constant factor in compute and rarely change its slope. Kaplan's exception is instructive. LSTMs match transformers on early tokens of the context but cannot use long contexts, so they scale worse (Figure 7). This is another bitter lesson: what matters is scale, not cleverness in the layers.
"""),

    # ------------------------------------------------------------------ §2
    markdown(r"""
<a id="wb5-section-2"></a>

## 2. Why power laws? A hundred unnecessary explanations

Very different mechanisms produce the same power-law learning curve, so the curve's form alone cannot tell you which one is at work.

""" + reading(
        "The quantization model of neural scaling",
        "https://arxiv.org/abs/2303.13506",
        "Michaud, Liu, Girit & Tegmark (NeurIPS 2023)",
        "Networks learn discrete skills ('quanta') in order of use frequency; Zipfian frequencies give power-law scaling. Decomposes model behaviour into candidate quanta using gradients.",
        importance=3,
        length=3,
    ) + reading(
        "On the asymptotic learning curves of kernel ridge regression under power-law decay",
        "https://arxiv.org/abs/2309.13337",
        "Li, Zhang & Lin (2023)",
        "Rigorous kernel-ridge learning curves for power-law spectra, including the saturation of Exercise 2.2. A rabbit hole.",
        importance=1,
        length=4,
    ) + r"""

### Exercises: power-law learning curves

Two toy models with the same answer.

**Exercise 2.1.** Two routes to one curve.

*Spectral model.* Features $x\in\mathbb R^\infty$ are Gaussian with covariance $\mathrm{diag}(\lambda_k)$, $\lambda_k=k^{-\alpha}$, $\alpha>1$. The target is $y=\sum_k\beta_kx_k+\varepsilon$, with noise variance $\sigma^2$, and we have $n$ samples. The excess loss of an estimate $\hat\beta$ is $\sum_k\lambda_k(\hat\beta_k-\beta_k)^2$. The signal power in mode $k$ is $c_k^2=\lambda_k\beta_k^2=k^{-\alpha'}$ with $\alpha'>1$ (a generic target has $\alpha'=\alpha$; smoother targets have $\alpha'>\alpha$).

*Quanta model.* A task consists of discrete skills $k=1,2,\dots$, used with frequencies $p_k\propto k^{-(1+a)}$, $a>0$. A model that has learned the $N$ most frequent skills has loss proportional to the frequency mass of the rest.

(a) Show that $\sum_{k>K}k^{-s}\sim K^{1-s}/(s-1)$ for $s>1$.

(b) *Heuristic for ridge.* Suppose the estimator learns modes $1,\dots,K$ with $K\le n$ and nothing beyond. Each learned mode costs about $\sigma^2/n$ (error variance $\sigma^2/(n\lambda_k)$, weighted by $\lambda_k$); each unlearned mode costs $c_k^2$. So
$$E(K)\approx\sum_{k>K}c_k^2+\frac{\sigma^2K}n .$$
Show that $E\sim n^{-(\alpha'-1)}$ without noise, and $E\sim n^{-(\alpha'-1)/\alpha'}$ with noise and the best $K$.

(c) Show that the quanta model's loss after learning $N$ skills scales as $N^{-a}$. If $N$ grows as a power of parameters or data, what form does the loss curve take?
"""),
    answer(r"""
**(a)** $k^{-s}$ is decreasing, so $\int_{K+1}^\infty t^{-s}dt\le\sum_{k>K}k^{-s}\le\int_K^\infty t^{-s}dt$. Both integrals are $\sim K^{1-s}/(s-1)$.

**(b)** With $s=\alpha'$, the tail is $\sim K^{-(\alpha'-1)}/(\alpha'-1)$. Without noise, take $K=n$: $E\sim n^{-(\alpha'-1)}$. With noise, $dE/dK=0$ gives $-K^{-\alpha'}+\sigma^2/n=0$, so $K^*=(n/\sigma^2)^{1/\alpha'}$, below $n$ for large $n$. Both terms are then of order $(K^*)^{1-\alpha'}$, so
$$E\sim(n/\sigma^2)^{-(\alpha'-1)/\alpha'} .$$

**(c)** By (a) with $s=1+a$, $\sum_{k>N}p_k\propto N^{-a}/a$. If $N\propto P^\mu$ for a resource $P$, the loss is $\propto P^{-a\mu}$, again a power law, with nothing spectral used. An observed power law cannot by itself tell the two mechanisms apart. Finer measurements can: per-example loss curves should be step-like under quanta and smooth under the spectral picture.
"""),
    markdown(r"""
**Exercise 2.2.** Where the heuristic fails: saturation.

Use the spectral model of Exercise 2.1. Ridge shrinks mode $k$ by the filter $\lambda_k/(\lambda_k+\kappa)$, so it also biases the modes it "learns". At large $n$ the effective regularization $\kappa$ sits near the eigenvalue of the last resolved mode; without noise take $\kappa\approx\lambda_n=n^{-\alpha}$. The bias from shrinking the head is
$$B_{\rm head}=\sum_{k<n}c_k^2\Big(\frac{\kappa}{\lambda_k+\kappa}\Big)^2 .$$

(a) Show that $B_{\rm head}\approx n^{-2\alpha}\sum_{k<n}k^{2\alpha-\alpha'}$ up to constants, and that modes with $k$ comparable to $n$ contribute at most the order of the tail, $n^{1-\alpha'}$.

(b) Evaluate the sum when $\alpha'-1<2\alpha$, when $\alpha'-1>2\alpha$, and at $\alpha'-1=2\alpha$. Conclude that the noiseless rate is $n^{-\min(\alpha'-1,\,2\alpha)}$ away from the boundary, and find the rate on it.

(c) With noise and optimally tuned ridge, redo Exercise 2.1(b) with $s=\min(\alpha'-1,2\alpha)$ in place of $\alpha'-1$, taking the cutoff $K$ as free. Show that $E\sim n^{-s/(s+1)}$ away from the boundary.

<details><summary>Hint</summary>

(a) For $k\le n/2$, $\lambda_k\ge2^\alpha\kappa$, so $\kappa/(\lambda_k+\kappa)$ is within a constant of $\kappa/\lambda_k=n^{-\alpha}k^\alpha$. For $n/2<k<n$ the filter factor is at most 1. (b) $\sum_{k<n}k^q$ grows like $n^{q+1}$ for $q>-1$, like $\log n$ for $q=-1$, and converges for $q<-1$.
</details>
"""),
    answer(r"""
**(a)** For $k\le n/2$, $\big(\kappa/(\lambda_k+\kappa)\big)^2\asymp n^{-2\alpha}k^{2\alpha}$, so the summand is $\asymp n^{-2\alpha}k^{2\alpha-\alpha'}$. For $n/2<k<n$ the factor is at most 1, and those modes contribute at most $\sum_{n/2<k<n}c_k^2\lesssim n^{1-\alpha'}$, the order of the tail.

**(b)** Let $q=2\alpha-\alpha'$.
- $\alpha'-1<2\alpha$ ($q>-1$): the sum is $\sim n^{q+1}$, so $B_{\rm head}\sim n^{-(\alpha'-1)}$, the order of the tail. The heuristic's exponent is right.
- $\alpha'-1>2\alpha$ ($q<-1$): the sum converges, so $B_{\rm head}\sim n^{-2\alpha}$, larger than the tail. Ridge **saturates** at $n^{-2\alpha}$, however smooth the target.
- $\alpha'-1=2\alpha$ ($q=-1$): the sum is $\sim\log n$, so $B_{\rm head}\sim n^{-2\alpha}\log n$.

**(c)** With cutoff $K$ ($\kappa\approx\lambda_K$), the same argument gives bias $\sim K^{-s}$ and variance $\sim\sigma^2K/n$. Minimizing gives $K^*\sim(n/\sigma^2)^{1/(s+1)}$ and $E\sim n^{-s/(s+1)}$. For $s=\alpha'-1$ this is Exercise 2.1(b). On the boundary there is an extra factor $(\log n)^{1/(s+1)}$.

These are heuristic scalings, valid up to constants. None of it needs the model to learn a good representation: plain ridge suffices. In both toy models the exponent is set by the data ($\alpha,\alpha'$, or $a$), except where the learner's own limits bind (ridge's saturation at $2\alpha$, or the quanta model's acquisition rate $\mu$). That is [§3](#wb5-section-3)'s theme.
"""),
    markdown(figure(SF, r"Panel B plots the deterministic-equivalent ridge risk (an asymptotic approximation) and Exercise 2.1's heuristic against $n$, with Monte Carlo ridge points. Toggle noise, and push $\alpha'$ past $2\alpha+1$ to see saturation.")),

    # ------------------------------------------------------------------ §3
    markdown(r"""
<a id="wb5-section-3"></a>

## 3. Another bitter lesson: architecture doesn't matter (much)

Since 2017, challengers to and refinements of the transformer have mostly moved the scaling curve by a constant factor in compute, rarely changed its slope, and where slopes measurably differ the alternatives usually scale *worse*.

""" + reading(
        "Deep learning scaling is predictable, empirically",
        "https://arxiv.org/abs/1712.00409",
        "Hestness et al. (2017)",
        "Power-law learning curves (against data, not compute) in translation, language, images and speech: \"model improvements only shift the error but do not appear to affect the power-law exponent.\"",
        importance=5,
        length=3,
    ) + reading(
        "Algorithmic progress in language models",
        "https://arxiv.org/abs/2403.05812",
        "Ho et al. (Epoch AI, 2024)",
        r"Estimates the transformer's compute-equivalent gain at about $7\times$ (95% CI roughly $3$–$46\times$), and required compute halving every 8 months or so. Treats each innovation as a constant multiplier, so cannot see slope changes.",
        importance=4,
        length=3,
    ) + reading(
        "Scaling laws vs model architectures: how does inductive bias influence scaling?",
        "https://arxiv.org/abs/2207.10551",
        "Tay et al. (2022)",
        "The counterpoint: compute slopes for about ten architectures differ, with the vanilla transformer steepest, so architectures that win small can lose at scale. Few points per fit.",
        importance=3,
        length=3,
    ) + reading(
        "Mamba: linear-time sequence modeling with selective state spaces",
        "https://arxiv.org/abs/2312.00752",
        "Gu & Dao (2023)",
        "The strongest recent challenger: a selective state-space model that matches a strong transformer recipe on roughly parallel scaling curves (Figure 4).",
        "Explainers: Grootendorst's [visual guide to Mamba](https://www.maartengrootendorst.com/blog/mamba/); [The annotated S4](https://srush.github.io/annotated-s4/).",
        importance=2,
        length=3,
    ) + reading(
        "I-JEPA: the first AI model based on Yann LeCun's vision for more human-like AI",
        "https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/",
        "Meta AI blog (2023; paper: [Assran et al.](https://arxiv.org/abs/2301.08243))",
        "A different objective rather than different layers: predict the representation of a masked region, not its pixels.",
        importance=2,
        length=1,
    ) + r"""
In vision, NFNet CNNs match vision transformers at equal compute (WB3).

### Exercises: what a constant factor is worth

**Exercise 3.1 (Fermi).** Write the reducible loss as $R(C)=A\,C^{-\alpha}$ in training compute $C$. Take $\alpha\approx0.05$ (Kaplan et al.'s exponent for loss against compute), and redo (a) with $\alpha\approx0.15$, roughly the Chinchilla fit's exponent for the reducible loss at compute-optimal allocation (WB3, scale section).

(a) Architecture B has $R_B(C)=(A/m)\,C^{-\alpha}$ with $m>1$: same exponent, better constant. Show that B at compute $C$ matches A at compute $m^{1/\alpha}C$ for every $C$. Ho et al. estimate the transformer's gain at about $7\times$ in compute. What reduction of reducible loss is that? What about the roughly $10^4\times$ of training compute separating GPT-2 from frontier models a few years later?

(b) B instead has a slightly better exponent: $R_B(C)=A\,C^{-(\alpha+\delta)}$, with $C$ in units of the compute where the curves cross. Show that B's compute multiplier at $C$ is $C^{\delta/\alpha}$. With $\delta/\alpha=0.1$ (a 10% better exponent), what is it three and ten orders of magnitude past the crossing?

(c) An architecture paper typically spans two or three decades of compute. Why does that make "a constant-factor gain" and "a better exponent" hard to tell apart, and which way would a small-scale advantage from a helpful inductive bias tend to mislead?
"""),
    answer(r"""
**(a)** $R_A(kC)=A\,k^{-\alpha}C^{-\alpha}$ equals $(A/m)C^{-\alpha}$ when $k=m^{1/\alpha}$, independent of $C$: on log–log axes B is A shifted left by $\log m/\alpha$. Conversely $m=k^\alpha$. With $\alpha=0.05$: $7^{0.05}\approx1.10$, a 10% lower reducible loss, and $10^{4\times0.05}\approx1.58$, 37% lower. With $\alpha=0.15$: $7^{0.15}\approx1.34$ (25%) and $10^{0.6}\approx4$ (75%). Because $\alpha$ is small, a modest loss improvement is a large compute multiplier, and four orders of magnitude would need a far larger constant-factor improvement than any architecture change has delivered.

**(b)** A reaches $A\,C^{-(\alpha+\delta)}$ at $C_A=C^{1+\delta/\alpha}$, so the multiplier is $C^{\delta/\alpha}$: $10^{0.3}\approx2$ three decades past the crossing and $10$ ten decades past it. A better exponent wins eventually, but slowly; over the explored range it looks like a constant factor.

**(c)** Over two or three decades a 10% exponent difference changes the multiplier by about $2\times$, comparable to fitting noise and to tuning one architecture better than another, so the two hypotheses give nearly the same plot. An inductive bias that suits the data helps most when data or compute is scarce, so it shows up as a small-scale advantage that shrinks: the better-looking architecture has the *shallower* slope. This is Tay et al.'s pattern, and extrapolating such a win is the mistake the bitter lesson warns against.
"""),

    # ------------------------------------------------------------------ §4
    markdown(r"""
<a id="wb5-section-4"></a>

## 4. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses

Different networks seem to learn similar representations, perhaps converging on one shared one whose concepts are directions; each claim means something only once you say which transformations of a representation count as cosmetic.

""" + reading(
        "Similarity of neural network representations revisited",
        "https://arxiv.org/abs/1905.00414",
        "Kornblith, Norouzi, Lee & Hinton (ICML 2019)",
        "Indices invariant to every invertible linear map (CCA included) go blind once width reaches the number of data points; proposes CKA, which matches corresponding layers across seeds.",
        importance=4,
        length=3,
    ) + reading(
        "The Platonic representation hypothesis",
        "https://arxiv.org/abs/2405.07987",
        "Huh, Cheung, Wang & Isola (ICML 2024)",
        "Representations across models and modalities align more as models get more capable (by mutual nearest neighbours); conjectures convergence to a shared statistical model of the world.",
        importance=5,
        length=3,
    ) + reading(
        "Proof of a perfect Platonic representation hypothesis",
        "https://arxiv.org/abs/2507.01098",
        "Ziyin & Chuang (2025)",
        "Deep linear networks of any depth and width, seeing different invertible views of the same noisy-label data, have every hidden layer aligned up to rotation and scale at the minima SGD's gradient noise selects. Elementary proof; needs label noise, and does not show SGD gets there.",
        importance=4,
        length=2,
    ) + reading(
        "The linear representation hypothesis and the geometry of large language models",
        "https://arxiv.org/abs/2311.03658",
        "Park, Choe & Veitch (ICML 2024)",
        "Makes \"concepts are directions\" precise with counterfactual pairs, and argues the geometry needs a chosen 'causal inner product' (Exercise 4.2).",
        importance=4,
        length=3,
    ) + reading(
        "Not all language model features are one-dimensionally linear",
        "https://arxiv.org/abs/2405.14860",
        "Engels, Michaud, Liao, Gurnee & Tegmark (ICLR 2025)",
        "The counterpoint: irreducibly multi-dimensional features, such as circles for days and months, used for modular arithmetic.",
        importance=3,
        length=3,
    ) + r"""
Also: [Neural thermodynamics](https://arxiv.org/abs/2505.12387), Ziyin, Xu & Chuang (2025), the same entropic-force argument applied more broadly. SVCCA, model stitching and superposition are in WB6.

### Exercises: similarity invariances

**Exercise 4.1.** Invariance groups, and why a minimum does not fix the representation.

Stack a layer's activations on $n$ inputs from two networks as rows of $X\in\mathbb R^{n\times p_1}$, $Y\in\mathbb R^{n\times p_2}$, columns centered, with Gram matrices $K=XX^\top$, $L=YY^\top$. Linear CKA is $\langle K,L\rangle_F/(\|K\|_F\|L\|_F)$. CCA's canonical correlations are the cosines of the principal angles between the column spaces of $X$ and $Y$.

(a) Show that $\langle K,L\rangle_F=\|Y^\top X\|_F^2$, and that CKA is unchanged by $X\mapsto XQ$ with $Q$ orthogonal and by $X\mapsto cX$, $c\neq0$. Show that it *is* changed by the stretch $A=\mathrm{diag}(2,1)$ applied to the four centered points $(\pm1,0),(0,\pm1)$.

(b) Show that CCA is unchanged by $X\mapsto XA$ for every invertible $A$. If centered $X$ and $Y$ both have rank $n-1$, show that all canonical correlations are 1, whatever the two networks compute.

(c) Take a deep linear network $f(x)=W_D\cdots W_1x$ and an invertible $T$. Show that $W_i\to TW_i$, $W_{i+1}\to W_{i+1}T^{-1}$ leaves $f$, and hence the loss, unchanged, while the layer-$i$ representation $h$ becomes $Th$. If the centered representations span $\mathbb R^{d_i}$, show that $\mathrm{CKA}(h,Th)=1$ only if $T$ is a scaled orthogonal matrix.

<details><summary>Hints</summary>

(a) Use cyclicity of the trace, and $\|K\|_F=\|X^\top X\|_F$.
(b) CCA depends only on column spaces; every centered column is orthogonal to $\mathbf 1$.
(c) CKA is 1 iff the Gram matrices are proportional: $h^\top T^\top Th'=c\,h^\top h'$ on a spanning set.
</details>
"""),
    answer(r"""
**(a)** $\langle K,L\rangle_F=\mathrm{Tr}(XX^\top YY^\top)=\mathrm{Tr}(Y^\top XX^\top Y)=\|Y^\top X\|_F^2$. Under $X\mapsto XQ$, $K\mapsto XQQ^\top X^\top=K$. Under $X\mapsto cX$, $K\mapsto c^2K$, which cancels. For the stretch, with $Y=X$ and $X'=XA$: $X^\top X=2I$, $X^\top X'=\mathrm{diag}(4,2)$ with squared norm $20$, $\|X'^\top X'\|_F=\|\mathrm{diag}(8,2)\|_F=2\sqrt{17}$, $\|X^\top X\|_F=2\sqrt2$. So $\mathrm{CKA}=20/(4\sqrt{34})=5/\sqrt{34}\approx0.86$. CKA's group is orthogonal maps and isotropic scaling, not all of $GL$.

**(b)** $\mathrm{col}(XA)=\mathrm{col}(X)$ for invertible $A$. Centered columns lie in $\mathbf 1^\perp$, of dimension $n-1$, so rank $n-1$ forces $\mathrm{col}(X)=\mathbf 1^\perp=\mathrm{col}(Y)$ and all principal angles vanish. This is Kornblith et al.'s point: invariance to every invertible linear map is too much for wide layers.

**(c)** $W_{i+1}T^{-1}TW_i=W_{i+1}W_i$, and the layer-$i$ output becomes $Th$. CKA is 1 iff $h^\top T^\top Th'=c\,h^\top h'$ for all pairs; if the $h$ span, bilinearity gives $T^\top T=cI$. For $d_i\ge2$, $T=\mathrm{diag}(2,1,\dots,1)$ is not scaled orthogonal. So every global minimum of a deep linear network sits on a $GL(d_i)$ orbit of global minima whose representations differ by CKA. Being a minimum fixes the function, not the representation; convergence of representations must come from whatever selects a point on the orbit, which is what Ziyin and Chuang analyse.
"""),
    markdown(r"""
**Exercise 4.2.** Which inner product?

A language model's next-token logits are $\lambda(x)^\top\gamma(y)$, where $\lambda(x)\in\mathbb R^d$ is the final hidden state for context $x$ and $\gamma(y)$ the unembedding of token $y$. Concept directions are differences such as $\gamma(y_1)-\gamma(y_0)$. Park et al.'s causal inner product is $\langle a,b\rangle_{\mathrm C}=a^\top\mathrm{Cov}(\gamma)^{-1}b$, with the covariance over the vocabulary (assume it invertible).

(a) Show that for any invertible $A$, $\lambda\to A\lambda$, $\gamma\to A^{-\top}\gamma$ leaves every logit, and so the model's behaviour, unchanged.

(b) Give two unembedding differences that are Euclidean-orthogonal before such a change and not after.

(c) Show that $\langle a,b\rangle_{\mathrm C}$, recomputed in the new coordinates, equals its old value.

<details><summary>Hint</summary>

(c) If $\gamma'=A^{-\top}\gamma$, then $\mathrm{Cov}(\gamma')=A^{-\top}\mathrm{Cov}(\gamma)A^{-1}$.
</details>
"""),
    answer(r"""
**(a)** $(A\lambda)^\top(A^{-\top}\gamma)=\lambda^\top A^\top A^{-\top}\gamma=\lambda^\top\gamma$.

**(b)** In $\mathbb R^2$, $a=e_1$, $b=e_2$ and $A^{-\top}=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ give $a'=(1,0)$, $b'=(1,1)$, with $a'\cdot b'=1$. Euclidean angles between concept directions are not fixed by the model's behaviour.

**(c)** $a'=A^{-\top}a$, $b'=A^{-\top}b$ and $\mathrm{Cov}(\gamma')^{-1}=A\,\mathrm{Cov}(\gamma)^{-1}A^\top$, so
$$a'^\top\mathrm{Cov}(\gamma')^{-1}b'=a^\top A^{-1}A\,\mathrm{Cov}(\gamma)^{-1}A^\top A^{-\top}b=a^\top\mathrm{Cov}(\gamma)^{-1}b .$$

Behaviour fixes the representation only up to $GL(d)$, the same group as Exercise 4.1(c), so a geometric claim needs extra structure. Whitening by the vocabulary covariance supplies an invariant geometry; whether it is the *right* one is what Park et al. argue from their assumption about causally separable concepts.
"""),
]


if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb5", "05_universality"))
