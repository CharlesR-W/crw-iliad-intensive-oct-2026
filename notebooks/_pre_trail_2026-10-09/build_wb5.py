#!/usr/bin/env python3
"""Build WB5 (Universality: the machines, they just want to learn).

Format: short reading notes plus a few derivation exercises with worked solutions.
Restructured 2026-10-09 at CRW's request: scaling laws open the workbook; the
architecture section is about scaling exponents; CKA/CCA, the PRH and the LRH are
one section with the papers linked rather than re-derived. Superposition and the
low-rank-logits / subliminal-learning material moved to WB6 (§5 and §13). The
pre-restructure builder, with the full perfect-PRH derivation, is kept in
_old_wb5/. Earlier record: reviews/wb5-editorial.md. Standard library only.
"""
from nbkit import answer, figure, markdown, reading, run

SF = "assets/interactive/spectral_filters.html"

CELLS = [
    # ------------------------------------------------------------------ intro
    markdown(r"""
# Workbook 5 — Universality: the machines, they just want to learn

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Different architectures trained on similar data end up with similar loss curves and, it seems, similar internal geometry. This workbook starts from the loss curves, asks how much architecture changes them, and then asks what "similar representations" can mean and why they might arise.

Route (the sections are largely independent):

- [1. Scaling laws: a hundred unnecessary explanations](#wb5-section-1). Exercises deriving power-law learning curves from a data spectrum and from discrete skills, including ridge saturation; interactive figure.
- [2. Another bitter lesson: architecture doesn't matter (much)](#wb5-section-2). Architectures move the scaling curve but rarely change its slope. One Fermi exercise on what a constant factor is worth; readings, with explainers for Mamba and JEPA.
- [3. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses](#wb5-section-3). What "same up to what" means for similarity indices, then the readings on representational convergence and linear concepts. Two short exercises.

Superposition, adversarial examples, low-rank logits and subliminal learning are in WB6 ([§5](06_mysteries_myriad.html#wb6-section-5) and [§13](06_mysteries_myriad.html#wb6-section-13)).

**How to use this notebook.** Pick what interests you; nobody is expected to do everything. Exercises are derivations with hints. Worked solutions are folded under "Solution" after each exercise, and everyone may use them. Each reading opens with a sentence or two on what it shows, so you can decide whether to open it, and is scored for importance and length (the legend under the contents explains the scales).
"""),

    # ------------------------------------------------------------------ §1
    markdown(r"""
<a id="wb5-section-1"></a>

## 1. Scaling laws: a hundred unnecessary explanations

Held-out loss falls as a power law in parameters, data and compute over many orders of magnitude (WB3 §5 fits these curves and uses them to allocate compute). Very different mechanisms produce the same power-law learning curve. The exercises derive one from a power-law data spectrum, with ridge regression, where extra target smoothness eventually stops helping, and one from discrete skills learned in order of frequency.

""" + reading(
        "The quantization model of neural scaling",
        "https://arxiv.org/abs/2303.13506",
        "Michaud, Liu, Girit & Tegmark (NeurIPS 2023)",
        "Proposes that networks learn discrete chunks of skill ('quanta') in order of how often they are used; with Zipfian use frequencies this gives power-law scaling in parameters and data. It uses language-model gradients to decompose model behaviour automatically into candidate quanta.",
        importance=3,
        length=3,
    ) + "\n\n" + reading(
        "On the asymptotic learning curves of kernel ridge regression under power-law decay",
        "https://arxiv.org/abs/2309.13337",
        "Li, Zhang & Lin (2023)",
        "Rigorous learning curves for kernel ridge regression with power-law eigenvalues, showing how the regularization, the target's smoothness and the noise set the rate.",
        "The rigorous counterpart of Exercise 1.2's heuristic, including saturation.",
        importance=1,
        length=4,
    ) + r"""

### 1.A. Setup

**Spectral model.** Features $x\in\mathbb R^\infty$ are Gaussian with covariance $\mathrm{diag}(\lambda_k)$, $\lambda_k=k^{-\alpha}$, $\alpha>1$. The target is $y=\sum_k\beta_kx_k+\varepsilon$ with noise variance $\sigma^2$. The excess population loss of an estimate $\hat\beta$ is $\sum_k\lambda_k(\hat\beta_k-\beta_k)^2$. The target's signal power in mode $k$ is
$$c_k^2=\lambda_k\beta_k^2=k^{-\alpha'},\qquad\alpha'>1 .$$
A generic target with $\beta_k$ of order 1 has $\alpha'=\alpha$; smoother targets, concentrated on the top modes, have $\alpha'>\alpha$. The language of spectral filters is WB2 §1.D's.

**Quanta model.** A task consists of discrete skills $k=1,2,\dots$, used with frequencies $p_k\propto k^{-(1+a)}$, $a>0$. A model that has learned the $N$ most frequent skills has loss proportional to the frequency mass of the rest.

**Exercise 1.1.** Two routes to one curve.

(a) Show that $\sum_{k>K}k^{-s}\sim K^{1-s}/(s-1)$ for $s>1$.

(b) *Heuristic for ridge.* Suppose the estimator learns modes $1,\dots,K$ with $K\le n$ and nothing beyond. Each learned mode costs about $\sigma^2/n$: its coefficient has error variance about $\sigma^2/(n\lambda_k)$, weighted by $\lambda_k$ in the loss. Each unlearned mode costs its signal $c_k^2$. So
$$E(K)\approx\sum_{k>K}c_k^2+\frac{\sigma^2K}n .$$
Show that $E\sim n^{-(\alpha'-1)}$ without noise, and that $E\sim n^{-(\alpha'-1)/\alpha'}$ with noise and the best $K$.

(c) Show that the quanta model's loss after learning $N$ skills scales as $N^{-a}$. If $N$ grows as a power of parameters or data, what form does the loss curve take?
"""),
    answer(r"""
**(a)** $k^{-s}$ is decreasing, so $\int_{K+1}^\infty t^{-s}dt\le\sum_{k>K}k^{-s}\le\int_K^\infty t^{-s}dt$. Both integrals are $\sim K^{1-s}/(s-1)$.

**(b)** With $s=\alpha'$, the tail is $\sim K^{-(\alpha'-1)}/(\alpha'-1)$. Without noise the variance term vanishes, so take $K$ as large as allowed, $K=n$: $E\sim n^{-(\alpha'-1)}$. With noise, set $dE/dK=0$: $-K^{-\alpha'}+\sigma^2/n=0$, so $K^*=(n/\sigma^2)^{1/\alpha'}$, which is below $n$ for large $n$. Both terms are then of order $K^*\cdot\sigma^2/n=(K^*)^{1-\alpha'}$, so
$$E\sim(K^*)^{-(\alpha'-1)}\sim(n/\sigma^2)^{-(\alpha'-1)/\alpha'} .$$

**(c)** By (a) with $s=1+a$, $\sum_{k>N}p_k\propto N^{-a}/a$. If $N\propto P^\mu$ for some resource $P$ (parameters or data), the loss is $\propto P^{-a\mu}$, again a power law. Nothing spectral was used. Two unrelated mechanisms give the same functional form, so an observed power law cannot by itself tell them apart. Distinguishing them needs finer measurements, such as per-example loss curves: one expects them to be step-like under quanta and smooth under the spectral picture.
"""),
    markdown(r"""
**Exercise 1.2.** Where the heuristic fails: saturation.

Ridge shrinks mode $k$ by the filter $\lambda_k/(\lambda_k+\kappa)$, so it also biases the modes it "learns". In the large-$n$ description of ridge, the effective regularization $\kappa$ sits around the eigenvalue of the last resolved mode. Without noise, take $\kappa\approx\lambda_n=n^{-\alpha}$. The bias from shrinking the head is
$$B_{\rm head}=\sum_{k<n}c_k^2\Big(\frac{\kappa}{\lambda_k+\kappa}\Big)^2 .$$

(a) Show that $B_{\rm head}\approx n^{-2\alpha}\sum_{k<n}k^{2\alpha-\alpha'}$ up to constants, and that modes with $k$ comparable to $n$ contribute at most the order of the tail, $n^{1-\alpha'}$.

(b) Evaluate the sum when $\alpha'-1<2\alpha$, when $\alpha'-1>2\alpha$, and at the boundary $\alpha'-1=2\alpha$. Conclude that the noiseless rate is $n^{-\min(\alpha'-1,\,2\alpha)}$ away from the boundary, and find the rate at the boundary.

(c) With noise and optimally tuned ridge, repeat Exercise 1.1(b) with $s=\min(\alpha'-1,2\alpha)$ in place of $\alpha'-1$, taking the cutoff $K$ as the free variable. Show that $E\sim n^{-s/(s+1)}$ away from the boundary, and that this reduces to Exercise 1.1(b) when $s=\alpha'-1$.

<details><summary>Hint</summary>

(a) For $k\le n/2$, $\lambda_k\ge2^\alpha\kappa$, so $\kappa/(\lambda_k+\kappa)$ is within a constant of $\kappa/\lambda_k=n^{-\alpha}k^\alpha$. For $n/2<k<n$ the filter factor is at most 1. (b) $\sum_{k<n}k^q$ grows like $n^{q+1}$ for $q>-1$, like $\log n$ for $q=-1$, and converges for $q<-1$.
</details>
"""),
    answer(r"""
**(a)** For $k\le n/2$, $\big(\kappa/(\lambda_k+\kappa)\big)^2\asymp(\kappa/\lambda_k)^2=n^{-2\alpha}k^{2\alpha}$, so $c_k^2(\cdot)^2\asymp n^{-2\alpha}k^{2\alpha-\alpha'}$. For $n/2<k<n$, the factor is at most 1, and those modes contribute at most $\sum_{n/2<k<n}c_k^2\lesssim n\cdot n^{-\alpha'}=n^{1-\alpha'}$, the order of the tail $\sum_{k\ge n}c_k^2$.

**(b)** Let $q=2\alpha-\alpha'$.
- $\alpha'-1<2\alpha$ ($q>-1$): the sum is $\sim n^{q+1}/(q+1)$, so $B_{\rm head}\sim n^{-2\alpha+2\alpha-\alpha'+1}=n^{-(\alpha'-1)}$, the same order as the tail. The rate is $n^{-(\alpha'-1)}$, and the heuristic's exponent is right.
- $\alpha'-1>2\alpha$ ($q<-1$): the sum converges, so $B_{\rm head}\sim n^{-2\alpha}$, which is larger than the tail $n^{-(\alpha'-1)}$. Ridge **saturates** at rate $n^{-2\alpha}$, however smooth the target. The heuristic's $n^{-(\alpha'-1)}$ is wrong here.
- $\alpha'-1=2\alpha$ ($q=-1$): the sum is harmonic, $\sim\log n$, so $B_{\rm head}\sim n^{-2\alpha}\log n$, a log factor above the tail $n^{-2\alpha}$.

So the noiseless rate is $n^{-\min(\alpha'-1,2\alpha)}$ away from the boundary, and $n^{-2\alpha}\log n$ on it.

**(c)** With effective cutoff $K$ (so $\kappa\approx\lambda_K$), the same argument with $n$ replaced by $K$ gives bias $\sim K^{-s}$ and variance $\sim\sigma^2K/n$. Minimizing $K^{-s}+\sigma^2K/n$ gives $K^*\sim(n/\sigma^2)^{1/(s+1)}$ and $E\sim n^{-s/(s+1)}$. For $s=\alpha'-1$ this is $n^{-(\alpha'-1)/\alpha'}$, as in Exercise 1.1(b). At the boundary the bias is $K^{-s}\log K$, and the same minimization gives an extra factor $(\log n)^{1/(s+1)}$.

These are heuristic scalings, valid up to constants and, near the boundary, logarithmic factors. Ridge risk in this model is described at large $n$ by deterministic-equivalent formulas, which are asymptotic approximations rather than the exact finite-$n$ risk. The rigorous rates, including saturation, are in Li, Zhang and Lin. None of this depends on the model learning a good representation: random features or plain ridge suffice. It accounts for the form of a learning curve, not for what is learned. In both models the exponent is set mainly by the data (the spectrum $\alpha,\alpha'$, or the skill frequencies $a$), except where the learner's own limits bind: ridge's saturation at $2\alpha$, or the rate $\mu$ at which the quanta model acquires skills. That is §2's theme.
"""),
    markdown(figure(SF, r"Use panel B. It plots the deterministic-equivalent ridge risk (an asymptotic approximation) and Exercise 1.1's heuristic against $n$, with the fitted slope, plus Monte Carlo ridge points at the default $\alpha,\alpha'$ (noise off, or $\sigma^2=10^{-2}$). Turn noise on and off, and push $\alpha'$ past $2\alpha+1$ to see saturation, where the heuristic's slope is wrong.")),

    # ------------------------------------------------------------------ §2
    markdown(r"""
<a id="wb5-section-2"></a>

## 2. Another bitter lesson: architecture doesn't matter (much)

Since 2017 there has been a long line of challengers to the transformer and refinements of it: state-space models such as Mamba, linear-attention and recurrent hybrids, mixtures of experts, the "Transformer++" recipe (rotary embeddings, SwiGLU, RMSNorm), and different objectives such as LeCun's JEPA. Many of them are real improvements. This section is about how big they are and what shape they have. On log–log scaling plots (WB3 §5) they mostly move the curve over by a roughly constant factor in compute, and rarely change its slope. A constant factor of a few, or even ten, is worth a lot of money. It is not the roughly four orders of magnitude of training compute that separate GPT-2 from the frontier models of a few years later. That came mostly from spending more compute.

**The evidence, briefly.**

- Hestness et al. fit learning curves against data size across translation, language modelling, image classification and speech. Their summary is that "model improvements only shift the error but do not appear to affect the power-law exponent." For example, LSTMs and recurrent highway networks have almost identical exponents. They suspect the exponent is set by the structure of the problem domain, which is also what both models of §1 give.
- In the Mamba paper's compute-scaling figure (perplexity against FLOPs on the Pile, 125M–1.3B parameters), the architectures fall on roughly parallel lines over about two decades of compute, which by Exercise 2.1(c) is too short to rule out a small slope difference. Mamba matches the Transformer++ recipe, and both sit below a GPT-3-style transformer by what reads off the plot as a few times in compute. (That multiplier is read off the plot; the paper does not state one.)
- Ho et al. estimate the compute-equivalent gain of the transformer itself at about $7\times$ (95% CI roughly $3$–$46\times$). That is about a fifth, on a log scale, of the roughly $2\times10^4$ total algorithmic progress they estimate for language modelling from 2014 to 2023 (a halving of required compute every 8 months or so; the total is an extrapolation, and it includes training recipes, data and compute allocation, not only architecture). They find that growth in compute contributed more still. Their model treats each innovation as a constant compute multiplier, so it could not detect a change in slope.
- In vision, NFNet CNNs match vision transformers at equal compute (WB3 §4).

**Where it fails.** "Same exponent" is an approximation, and the exceptions are instructive. Kaplan et al.'s LSTMs scale visibly worse than transformers with model size. The gap comes from later tokens in the context: LSTMs match transformers early in the context but cannot use long contexts. In effect the architecture changes which part of the data distribution is learnable at all, so it is not the same data spectrum. Tay et al. fit compute slopes for about ten architectures and find them genuinely different, with the vanilla transformer steepest. So the slope differences that have been measured mostly show alternatives scaling *worse*, and architectures that look good at small scale can fall behind. A defensible summary is that nobody has clearly beaten the transformer's exponent on language, and that the measured improvements are constant factors of roughly one order of magnitude at most.

""" + reading(
        "Deep learning scaling is predictable, empirically",
        "https://arxiv.org/abs/1712.00409",
        "Hestness et al. (2017)",
        "Measures power-law learning curves in four domains and finds that architecture changes move the curve down without changing its exponent, which seems to be a property of the domain. "
        "Note that these are curves against training-set size, not compute.",
        importance=5,
        length=3,
    ) + "\n\n" + reading(
        "Algorithmic progress in language models",
        "https://arxiv.org/abs/2403.05812",
        "Ho, Besiroglu, Erdil, Owen, Rahman, Guo, Atkinson, Thompson & Sevilla (Epoch AI, 2024)",
        "Fits augmented scaling laws to 200+ language-model evaluations to separate the contributions of compute and algorithms. "
        "The source of the $7\times$ transformer and 8-month halving figures above, with their caveats (context length was not controlled). "
        "[Epoch's summary page](https://epoch.ai/publications/algorithmic-progress-in-language-models).",
        importance=4,
        length=3,
    ) + "\n\n" + reading(
        "Mamba: linear-time sequence modeling with selective state spaces",
        "https://arxiv.org/abs/2312.00752",
        "Gu & Dao (2023)",
        "A recurrent state-space model with input-dependent ('selective') dynamics that trains in parallel and matches a strong transformer recipe on language at the scales tested. Figure 4 is the scaling comparison discussed above.",
        "To understand the architecture first, read Grootendorst's [A visual guide to Mamba and state space models](https://www.maartengrootendorst.com/blog/mamba/); for the mathematics of the S4 layer underneath it, Rush and Karamcheti's [The annotated S4](https://srush.github.io/annotated-s4/).",
        importance=2,
        length=3,
    ) + "\n\n" + reading(
        "I-JEPA: the first AI model based on Yann LeCun's vision for more human-like AI",
        "https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/",
        "Meta AI blog (2023; paper: [Assran et al.](https://arxiv.org/abs/2301.08243))",
        "An accessible introduction to joint-embedding predictive architectures, which predict the *representation* of a masked region rather than its pixels. "
        "JEPA changes the objective rather than the layers; WB3 §2 discusses the argument for it, and LeCun's position paper.",
        importance=2,
        length=1,
    ) + "\n\n" + reading(
        "Scaling laws vs model architectures: how does inductive bias influence scaling?",
        "https://arxiv.org/abs/2207.10551",
        "Tay et al. (2022)",
        "Fits compute-scaling slopes for about ten architectures (Switch, Performer, MLP-Mixer, ALBERT and others) and finds them distinct, with the vanilla transformer steepest; models that do well at one scale need not at another. "
        "Fits use few points per architecture.",
        importance=3,
        length=3,
    ) + "\n\n" + reading(
        "Scaling laws for neural language models",
        "https://arxiv.org/abs/2001.08361",
        "Kaplan et al. (2020)",
        "Loss depends strongly on parameters, data and compute, and only weakly on shape: the aspect ratio can vary by a factor of 40 at little cost. "
        "Figure 7 is the LSTM comparison above. WB3 §5 uses the same paper for compute allocation.",
        importance=2,
        length=3,
    ) + r"""

**Exercise 2.1 (Fermi).** What a constant factor is worth. Write the reducible loss as $R(C)=A\,C^{-\alpha}$. Take $\alpha\approx0.05$, Kaplan et al.'s exponent for loss against compute, and redo (a) with $\alpha\approx0.15$, roughly what the Chinchilla fit (WB3 §5) gives for the reducible part at compute-optimal allocation: $\alpha\beta/(\alpha+\beta)$ with their $0.34$ and $0.28$. Kaplan's fit has no irreducible term, so $0.05$ is the exponent of the whole loss.

(a) Architecture B has $R_B(C)=(A/m)\,C^{-\alpha}$ with $m>1$: the same exponent, a better constant. Show that B at compute $C$ matches A at compute $m^{1/\alpha}C$ for every $C$. What reduction of reducible loss corresponds to the transformer's $7\times$? To four orders of magnitude of compute?

(b) Architecture B instead has a slightly better exponent: $R_B(C)=A\,C^{-(\alpha+\delta)}$, with $C$ measured in units of the compute at which the two curves cross. Show that B's compute multiplier at compute $C$ is $C^{\delta/\alpha}$. With $\delta=0.005$ (a 10% better exponent), what is the multiplier three and ten orders of magnitude past the crossing?

(c) An architecture paper typically spans two or three decades of compute. Why does that make "a constant-factor gain" and "a better exponent" hard to tell apart, and which way would a small-scale advantage from a helpful inductive bias tend to mislead?
"""),
    answer(r"""
**(a)** $R_A(kC)=A\,k^{-\alpha}C^{-\alpha}$ equals $R_B(C)=(A/m)C^{-\alpha}$ when $k^{-\alpha}=1/m$, that is $k=m^{1/\alpha}$, independently of $C$. So on log–log axes B's curve is A's shifted left by $\log m/\alpha$. Conversely $m=k^{\alpha}$. For $k=7$: $m=7^{0.05}=e^{0.05\ln7}\approx1.10$, a 10% lower reducible loss at every scale. For $k=10^4$: $m=10^{0.2}\approx1.58$, a 37% lower reducible loss at every scale. Because $\alpha$ is small, a modest-looking loss improvement is a large compute multiplier. With $\alpha=0.15$: $7^{0.15}\approx1.34$, a 25% reduction, and $10^{0.6}\approx4$, a 75% reduction. Either way, four orders of magnitude would need a constant-factor improvement much larger than any architecture change has delivered.

**(b)** B at compute $C$ reaches $A\,C^{-(\alpha+\delta)}$. A reaches this at $C_A$ with $C_A^{-\alpha}=C^{-(\alpha+\delta)}$, so $C_A=C^{1+\delta/\alpha}$ and the multiplier is $C_A/C=C^{\delta/\alpha}$. With $\delta/\alpha=0.1$: $10^{0.3}\approx2$ three decades past the crossing, and $10^{1}=10$ ten decades past it. A better exponent wins without bound eventually, but slowly. Over the range that has actually been explored, its effect is about the size of a constant factor.

**(c)** Over two or three decades, a 10% exponent difference changes the multiplier by a factor of about two, which is comparable to fitting noise and to the effect of tuning one architecture better than another. A constant factor and a small slope difference therefore produce nearly the same plot. An inductive bias that suits the data helps most when data or compute is scarce (WB3 §4), so it shows up as an advantage at small scale that then shrinks: the curves converge, so the better-looking architecture has the *shallower* slope. Tay et al.'s finding that alternatives look good small and fall behind at scale is this pattern. Extrapolating a small-scale win is the mistake the bitter lesson warns against.
"""),

    # ------------------------------------------------------------------ §3
    markdown(r"""
<a id="wb5-section-3"></a>

## 3. Same representation? CKA, CCA, and the Platonic and linear representation hypotheses

Three claims tend to get run together: that the internal representations of different networks are measurably similar; that, as models get more capable, they converge toward one shared representation (the Platonic representation hypothesis, PRH); and that the concepts in that representation are directions (the linear representation hypothesis, LRH). Each claim means something only once you say which transformations of a representation count as cosmetic. This section gives that vocabulary in two short exercises and otherwise points you to the papers.

### 3.A. Same up to what?

Take $n$ inputs. Stack a layer's activations from two networks as the rows of $X\in\mathbb R^{n\times p_1}$ and $Y\in\mathbb R^{n\times p_2}$, and center the columns. Write $K=XX^\top$ and $L=YY^\top$ for the Gram matrices, whose entries are inner products between inputs. Three common indices, each with its own invariance group:

- **Linear CKA** (centered kernel alignment), $\mathrm{CKA}(X,Y)=\langle K,L\rangle_F/(\|K\|_F\|L\|_F)$. It equals 1 exactly when $K\propto L$, that is, when the two representations agree up to an isometry between the subspaces they occupy, a positive scale and a translation.
- **CCA** (canonical correlation analysis): the canonical correlations are the cosines of the principal angles between the column spaces of $X$ and $Y$ in $\mathbb R^n$. SVCCA (WB6 §2) truncates each representation to its top singular directions first.
- **Mutual $k$-nearest neighbours**, the index Huh et al. use in §3.B: do the two representations agree on each input's $k$ nearest neighbours?

Networks also have exact parameter symmetries. Permuting the hidden units of a layer, together with the matching columns of the next layer's weights, leaves the function unchanged, giving $\prod_\ell n_\ell!$ equivalent settings; for ReLU units, scaling a unit's incoming weights by $c>0$ and its outgoing weights by $1/c$ does too. Permutations are invisible to all three indices. Exercise 3.1(c) is the continuous version for linear networks, and there the indices do see the difference.

""" + reading(
        "Similarity of neural network representations revisited",
        "https://arxiv.org/abs/1905.00414",
        "Kornblith, Norouzi, Lee & Hinton (ICML 2019)",
        "Shows that similarity indices invariant to all invertible linear maps, CCA included, cannot distinguish representations once the width reaches the number of data points. Proposes CKA, which identifies corresponding layers across networks trained from different initializations.",
        "Exercise 3.1 derives the invariance groups. Model stitching (WB6 §3) is the functional alternative: can one network's layer be used by another's upper layers?",
        importance=4,
        length=3,
    ) + r"""

**Exercise 3.1.** Invariance groups, and why a minimum does not fix the representation.

(a) Show that $\langle K,L\rangle_F=\|Y^\top X\|_F^2$, and that CKA is unchanged by $X\mapsto XQ$ with $Q$ orthogonal and by $X\mapsto cX$, $c\neq0$. Show that it *is* changed by the stretch $A=\mathrm{diag}(2,1)$ applied to the four centered points $(\pm1,0),(0,\pm1)$.

(b) Show that CCA is unchanged by $X\mapsto XA$ for every invertible $A$. If the centered $X$ and $Y$ both have rank $n-1$ (possible once the width is at least $n-1$), show that all canonical correlations are 1, whatever the two networks compute.

(c) Take a deep linear network $f(x)=W_D\cdots W_1x$ and an invertible $T$. Show that $W_i\to TW_i$, $W_{i+1}\to W_{i+1}T^{-1}$ leaves $f$, and hence the loss, unchanged, while the layer-$i$ representation $h$ becomes $Th$. If the centered representations span $\mathbb R^{d_i}$, show that $\mathrm{CKA}(h,Th)=1$ only if $T$ is a scaled orthogonal matrix. (So for $d_i\ge2$ some minima on this orbit have different representations.)

<details><summary>Hints</summary>

(a) Use cyclicity of the trace. With $Y=X$ and $X'=XA$, $\mathrm{CKA}=\|X^\top X'\|_F^2/(\|X'^\top X'\|_F\|X^\top X\|_F)$, using $\|K\|_F=\|X^\top X\|_F$.
(b) CCA depends only on the column spaces; every column of a centered matrix is orthogonal to $\mathbf 1$.
(c) CKA is 1 iff the Gram matrices are proportional: $h^\top T^\top Th'=c\,h^\top h'$ on a spanning set. A bilinear form that vanishes on a spanning set vanishes everywhere.
</details>
"""),
    answer(r"""
**(a)** $\langle K,L\rangle_F=\mathrm{Tr}(XX^\top YY^\top)=\mathrm{Tr}(Y^\top XX^\top Y)=\|Y^\top X\|_F^2$. Under $X\mapsto XQ$, $K=XX^\top\mapsto XQQ^\top X^\top=K$. Under $X\mapsto cX$, $K\mapsto c^2K$, and the factor cancels between numerator and denominator. For the stretch: $X^\top X=2I$, $X^\top X'=2A=\mathrm{diag}(4,2)$ with squared norm $20$, $X'^\top X'=2A^2=\mathrm{diag}(8,2)$ with norm $2\sqrt{17}$, and $\|X^\top X\|_F=2\sqrt2$. So $\mathrm{CKA}=20/(4\sqrt{34})=5/\sqrt{34}\approx0.86$. CKA's group is orthogonal maps and isotropic scaling (plus translation, removed by centering), not all of $GL$.

**(b)** $\mathrm{col}(XA)=\mathrm{col}(X)$ for invertible $A$. Centered columns lie in $\mathbf 1^\perp$, of dimension $n-1$, so rank $n-1$ means $\mathrm{col}(X)=\mathbf 1^\perp=\mathrm{col}(Y)$: all principal angles vanish. This is Kornblith et al.'s argument that invariance to every invertible linear map is too much invariance for wide layers.

**(c)** $W_{i+1}T^{-1}TW_i=W_{i+1}W_i$, so $f$ is unchanged, and the layer-$i$ output is $TW_i\cdots W_1x=Th$. CKA is 1 iff $h^\top T^\top Th'=c\,h^\top h'$ for all pairs of centered representations; if these span, bilinearity gives $T^\top T=cI$. For $d_i\ge2$ there are invertible $T$ that are not scaled orthogonal, e.g. $\mathrm{diag}(2,1,\dots,1)$, and at a global minimum the layer's representations span whenever its width equals the rank of the end-to-end map. So every global minimum of a deep linear network sits on a $GL(d_i)$ orbit of global minima whose representations are *not* all the same by CKA (or by nearest neighbours). Being a minimum fixes the function, not the representation. Any convergence of representations has to come from whatever selects a point on the orbit, which is what Ziyin and Chuang's paper (§3.B) is about.
"""),
    markdown(r"""
### 3.B. The Platonic representation hypothesis

Huh et al. observe that representations of different models, and even of models of different modalities, become more aligned as the models become more capable, and conjecture that they are converging toward a shared statistical model of the world.

""" + reading(
        "The Platonic representation hypothesis",
        "https://arxiv.org/abs/2405.07987",
        "Huh, Cheung, Wang & Isola (ICML 2024)",
        "Reports that representations in different models, architectures and modalities become more aligned as the models become more capable, measured mainly by mutual nearest neighbours. Hypothesizes convergence toward a shared statistical model of the world, driven by three pressures: multitask scaling, capacity and a simplicity bias.",
        "It discusses limitations, including information specific to one modality that cannot converge.",
        importance=5,
        length=3,
    ) + "\n\n" + reading(
        "Proof of a perfect Platonic representation hypothesis",
        "https://arxiv.org/abs/2507.01098",
        "Ziyin & Chuang (2025)",
        "Two deep linear networks of any depths, with hidden widths at least the rank of the target map, wrapped in arbitrary fixed invertible maps and trained on different invertible views of the same noisy-label data, have every pair of hidden layers aligned up to rotation and scale at the minima selected by SGD's small-learning-rate modified objective.",
        "Short, and the proof is elementary linear algebra; the paper also lists the ways the result breaks. It does not prove that SGD converges to these minima.",
        importance=4,
        length=2,
    ) + r"""

**What Ziyin and Chuang show, in outline.** The setting: underlying inputs $x$ with second moment $\Sigma_x$, and labels $y=V^*x+\epsilon$ with noise covariance $\Sigma_\epsilon$. Network $A$ sees $x$ through an invertible "modality" map $Z_A$, network $B$ through $Z_B$, and each computes $M^OW_D\cdots W_1M^I$ on its input, with fixed invertible $M^I,M^O$ standing in for the rest of an architecture; both are trained with squared loss on the same labels. Exercise 3.1(c) says a global minimum of a deep linear network does not fix its representations. Backward-error analysis of small-step SGD says training approximately minimizes $L+\eta S$, where $S=\mathbb E\|\nabla_\theta\ell\|^2$ is, at a global minimum, the variance of the per-sample gradients. As $\eta\to0^+$ this picks, among global minima, those with the least gradient noise. Moving along the orbit of Exercise 3.1(c) changes $S$, and stationarity along it forces a *balance condition* between adjacent layers. In whitened coordinates (multiply by $\Sigma_\epsilon^{1/2}$ on the output side and $\Sigma_x^{1/2}$ on the input side, where $\Sigma_\epsilon$ is the label-noise covariance) the balance condition and the end-to-end constraint have a solution unique up to rotation and scale, and every hidden layer's Gram matrix comes out proportional to
$$x^\top\Sigma_x^{-1/2}\big(\bar V^\top\bar V\big)^{1/2}\Sigma_x^{-1/2}x',\qquad \bar V=\Sigma_\epsilon^{1/2}V^*\Sigma_x^{1/2},$$
in terms of the underlying input $x$. Nothing here mentions the fixed input and output maps, the input "modality", the depth, the width or which layer it is. So every layer of every such network is perfectly aligned (CKA $=1$).

Things worth noticing before you lean on it:

- It characterizes minimizers of $S$ among global minima; it does not show SGD reaches them.
- The selection needs label noise. With $\Sigma_\epsilon=0$, every per-sample gradient vanishes at every global minimum, so $S=0$ on all of them and nothing is selected.
- Gradient flow conserves $W_{i+1}^\top W_{i+1}-W_iW_i^\top$, so without the noise term a limit point, if training converges, keeps an imprint of the initialization.
- Weight decay (MAP under a Gaussian prior) selects a different, norm-balanced point, and in the simplest example its representation depends on the input modality. So the gradient-noise selection and the MAP one disagree here (compare WB2 §4.B).
- Training on invertibly transformed labels $\Phi y$ breaks alignment unless $\Phi$ is a scaled orthogonal matrix: the shared representation depends on the labels through both $V^*$ and $\Sigma_\epsilon$.
- None of Huh et al.'s multitask or capacity pressures appears: there is one task, and the selection is an implicit bias acting on a parameter symmetry.

""" + reading(
        "Neural thermodynamics: entropic forces in deep and universal representation learning",
        "https://arxiv.org/abs/2505.12387",
        "Ziyin, Xu & Chuang (NeurIPS 2025)",
        "The same entropic-force argument applied more broadly, including to sharpness- and flatness-seeking behaviour of SGD.",
        importance=2,
        length=3,
    ) + r"""

The Training Dynamics materials develop the dynamics behind the selection.

### 3.C. The linear representation hypothesis

The hypothesis says that high-level concepts are directions in a model's representation space. One reading makes it precise and argues that its geometry needs a chosen inner product; the other finds features that are irreducibly multi-dimensional. Superposition, which asks how many such directions fit, is in WB6 §5.

""" + reading(
        "The linear representation hypothesis and the geometry of large language models",
        "https://arxiv.org/abs/2311.03658",
        "Park, Choe & Veitch (ICML 2024)",
        "Formalizes the hypothesis with counterfactual pairs, in the unembedding space (linked to linear probing) and in the embedding space (linked to steering). Shows that a non-Euclidean 'causal inner product', under which causally separable concepts are orthogonal, unifies the two; LLaMA-2 experiments find such representations and show that the choice of inner product matters.",
        "Exercise 3.2 works through the non-identifiability that makes the inner product a choice.",
        importance=4,
        length=3,
    ) + "\n\n" + reading(
        "Not all language model features are one-dimensionally linear",
        "https://arxiv.org/abs/2405.14860",
        "Engels, Michaud, Liao, Gurnee & Tegmark (ICLR 2025)",
        "Uses sparse autoencoders on GPT-2 and Mistral 7B to find features that cannot be split into independent or non-co-occurring lower-dimensional parts, most clearly circles for days of the week and months of the year. Intervention experiments show that models use these circles for modular arithmetic on days and months.",
        importance=3,
        length=3,
    ) + r"""

A language model's next-token logits are $\lambda(x)^\top\gamma(y)$, where $\lambda(x)\in\mathbb R^d$ is the final hidden state for context $x$ and $\gamma(y)$ is the unembedding vector of token $y$. Concept directions are differences such as $\gamma(y_1)-\gamma(y_0)$ over counterfactual pairs. Park et al. estimate their causal inner product as $\langle a,b\rangle_{\mathrm C}=a^\top\mathrm{Cov}(\gamma)^{-1}b$, with the covariance over the vocabulary's unembedding vectors (assume it is invertible).

**Exercise 3.2.** Which inner product?

(a) Show that for any invertible $A$, the change $\lambda\to A\lambda$, $\gamma\to A^{-\top}\gamma$ leaves every logit, and hence the model's behaviour, unchanged.

(b) Give two unembedding differences that are orthogonal in the Euclidean inner product before such a change and not after it.

(c) Show that $\langle a,b\rangle_{\mathrm C}$, recomputed in the new coordinates (with $a,b$ transformed as unembedding differences and the covariance recomputed), equals its old value.

<details><summary>Hint</summary>

(c) If $\gamma'=A^{-\top}\gamma$, then $\mathrm{Cov}(\gamma')=A^{-\top}\mathrm{Cov}(\gamma)A^{-1}$.
</details>
"""),
    answer(r"""
**(a)** $(A\lambda)^\top(A^{-\top}\gamma)=\lambda^\top A^\top A^{-\top}\gamma=\lambda^\top\gamma$.

**(b)** In $\mathbb R^2$, take $a=e_1$, $b=e_2$ and $A^{-\top}=\begin{pmatrix}1&1\\0&1\end{pmatrix}$. Then $a'=(1,0)$ and $b'=(1,1)$, with $a'\cdot b'=1\neq0$. Euclidean orthogonality, and cosine similarity in general, of concept directions is not determined by the model's input–output behaviour.

**(c)** $a'=A^{-\top}a$, $b'=A^{-\top}b$ and $\mathrm{Cov}(\gamma')^{-1}=A\,\mathrm{Cov}(\gamma)^{-1}A^\top$, so
$$a'^\top\mathrm{Cov}(\gamma')^{-1}b'=a^\top A^{-1}A\,\mathrm{Cov}(\gamma)^{-1}A^\top A^{-\top}b=a^\top\mathrm{Cov}(\gamma)^{-1}b .$$

This is §3.A's "same up to what" again. The model's behaviour fixes the representation only up to $GL(d)$, the same group as Exercise 3.1(c), and a geometric claim needs more structure. Whitening by the vocabulary's covariance, $g(y)=\mathrm{Cov}(\gamma)^{-1/2}(\gamma(y)-\mathbb E\gamma)$, turns $\langle\cdot,\cdot\rangle_{\mathrm C}$ into the Euclidean inner product and gives a geometry invariant under $A$, up to an orthogonal transformation. Whether it is the *right* geometry is what Park et al. argue from their assumption about causally separable concepts; read the paper for that step.
"""),

    # ------------------------------------------------------------------ closing
    markdown(r"""
## Related workbooks and further study

- WB3 §4 (ConvNets matching ViTs at scale) and WB3 §5 (fitting and using scaling laws) are the empirical background to §1–2.
- WB6 §2–3 (SVCCA, model stitching) are experimental counterparts to §3.A; WB6 §5 (superposition, adversarial examples) and §13 (low-rank logits, subliminal learning) are further evidence of shared structure across models.
- The Training Dynamics materials develop the dynamics behind §3.B's selection.
"""),
]


if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb5", "05_universality"))
