"""WB2 opening and §1 keystone: vacuous bounds and double descent.

Exercises 1.3-1.6 are checked by checks/wb2a_check.py (old numbering 3.A-3.C).
"""
from nbkit import *  # noqa: F401,F403

OPENING = [
    markdown(
        r"""
# Workbook 2 — How Deep Learning Changed Statistical Learning Theory

**Mysteries of Deep Learning, Iliad Intensive 2026**

Classical learning theory suggests that a model with sufficient capacity to
fit its training data exactly has no reason to generalize well on fresh data.
Deep networks, however, can generalize well while being highly
overparameterized and fitting their training data exactly. This workbook
explores what these observations broke in learning theory, and what survives.
It is the derivations workbook, so most sections end in exercises with worked
solutions folded beneath them.

- [1. Keystone: vacuous bounds and double descent](#wb2-keystone): random labels make class-wide bounds vacuous; past interpolation, test error falls again.
- [2. Reference cases: risk, least squares, bias–variance](#wb2-section-2): the textbook quantities, computed exactly.
- [3. Theorems that seem to forbid learning](#wb2-section-3): no free lunch, universal approximation, VC dimension.
- [4. Regularization: prior, constraint, filter, flatness](#wb2-section-4): what weight decay, early stopping, dropout and flat minima do in linear models.
- [5. Interpolation without harm](#wb2-section-5): kernels, the ridgeless limit, benign overfitting.
- [6. PAC-Bayes](#wb2-section-6): a bound for the function you found, not the class.
- [7. Bayes and MLE generalization error](#wb2-section-7): the sandwich, TIC, AIC, BIC; a fit versus the Bayes ensemble.
- [8. Gradient noise, curvature, and generalization measures](#wb2-section-8): SGD is not a posterior sampler; which measures predict the gap.
- [9. The information bottleneck](#wb2-section-9): did deep networks compress?

Exercise $k.m$ part $j$ is cited as $k.m.j$. Each exercise states the notation it uses.
"""
    ),
]

KEYSTONE = [
    markdown(
        r"""
<a id="wb2-keystone"></a>

## 1. Keystone: vacuous bounds and double descent

Two observations broke the textbook account of generalization:

1. Standard networks fit uniformly random labels to zero training error
   (Zhang et al.), so any bound on generalization that depends only on the
   model class must be vacuous.
2. As model size grows through the interpolation threshold (where training
   error first reaches zero), test error falls, rises, and falls again
   (Belkin et al.), so fitting the training data exactly is not by itself
   overfitting.

Every later section of this workbook is a response to one of these
observations. The readings below give an overview of these observations.
Readers should feel free to read them in as much depth as their interest and
time permit, as guided by the marked importance and length.

Notation (used throughout the workbook): $n$ samples, $p$ parameters or features, $\gamma=p/n$; $\sigma^2$ is the label-noise variance, $\tau^2$ a Gaussian prior variance on the coefficients $\beta$, $r^2=\|\beta\|^2$, and $\lambda$ the ridge penalty in $\frac1n\|y-X\beta\|^2+\lambda\|\beta\|^2$.
"""
        + reading(
            "The paper that killed deep learning theory",
            "https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory",
            "Lawrence Chan (LessWrong, 2026)",
            "A short account of Zhang et al. (2016), and why fitting random labels ended the programme of "
            "explaining generalization by the capacity of the model class.",
            importance=5, length=1,
        )
        + reading(
            "The other paper that killed deep learning theory",
            "https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory",
            "Lawrence Chan (LessWrong, 2026)",
            "On Nagarajan and Kolter (2019): the data-dependent norm and margin bounds proposed after Zhang et "
            "al. grow with the sample size, and uniform convergence provably cannot explain generalization in "
            "some overparametrized settings, even restricted to what gradient descent outputs.",
            importance=4, length=1,
        )
        + reading(
            "Understanding deep learning requires rethinking generalization",
            "https://arxiv.org/abs/1611.03530",
            "Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals (ICLR 2017)",
            "Standard image classifiers trained by SGD reach zero training error on random labels and random "
            "pixels, with or without explicit regularization. Read §1–§2; §3 is short.",
            importance=5, length=2,
        )
        + reading(
            "Reconciling modern machine-learning practice and the classical bias–variance trade-off",
            "https://arxiv.org/abs/1812.11118",
            "Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal (PNAS, 2019)",
            "Names double descent: test error peaks at the interpolation threshold and falls again as capacity "
            "grows, in random-feature models, small networks and tree ensembles. The figures carry the argument.",
            importance=4, length=2,
        )
        + reading(
            "Maybe I was too harsh on deep learning theory (three days ago)",
            "https://www.lesswrong.com/posts/6SRq7mZ97Dwuavwb6/maybe-i-was-too-harsh-on-deep-learning-theory-three-days-ago",
            "Lawrence Chan (LessWrong, 2026)",
            "A partial retraction: the signal-propagation (mean-field) and Tensor Programs line, which produced "
            "μP, counts as real cumulative theory, though it does not explain generalization.",
            importance=2, length=1,
        )
        + r"""
Also: [Nagarajan and Kolter, *Uniform convergence may be unable to explain generalization in deep learning*](https://arxiv.org/abs/1902.04742) (NeurIPS 2019); [Nakkiran et al., *Deep double descent*](https://arxiv.org/abs/1912.02292) (2019), double descent in model size, training time and sample size for real networks.

### Exercises: capacity bounds meet a real network

This exercise makes observation 1 above quantitative. Classical
generalization bounds depend only on the function class, the sample size $n$
and a confidence level, and so in particular must hold for every data
distribution. Zhang et al. trained standard architectures on CIFAR-10 images
with labels drawn uniformly at random. Despite the mislabeling, the networks
still reached zero training error, while every classifier has expected test
error 0.9 on such data (there are ten classes, and the labels carry no
information about the image). Because the functions reachable by training
include ones with a test-train gap of 0.9, a bound that holds for every
distribution must allow that gap on the real labels too, where the same
networks are known to generalize well. Classical generalization bounds are
therefore vacuous for these networks. Nagarajan and Kolter showed that
shrinking the function class does not help: they construct data distributions
on which gradient descent generalizes well, yet any uniform bound over only the
functions gradient descent returns on samples from that distribution is still
vacuous, even though such a bound may depend on both the distribution and the
training procedure.

**1.1. Plug in a real network.** Use the 0–1 loss, $\ell(h(x),y)=\mathbf 1[h(x)\ne y]$,
which is 1 on a mistake and 0 otherwise. The *population risk* of a classifier
$h$ is its expected loss on a fresh example drawn from the data distribution,
$R(h)=\mathbb E[\ell(h(x),y)]$; for the 0–1 loss this is its test error. The
*empirical risk* $\hat R_n(h)$ is its average loss over the $n$ training
examples, its training error. The difference $R(h)-\hat R_n(h)$ is the
*generalization gap*, and it is what a generalization bound controls. Call the
set of classifiers the learner may choose from the *hypothesis class*
$\mathcal H$. The simplest of the classical bounds applies when
$\mathcal H$ is finite: with probability at least $1-\delta$ over the draw of
the training set, every $h\in\mathcal H$ at once satisfies
$|R(h)-\hat R_n(h)|\le\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$. "Every $h$ at
once" is what makes this a *uniform* bound; the bounds for infinite classes
(§3) have the same form with $\ln|\mathcal H|$ replaced by another measure of
the size of the class. A bound is *vacuous* if it exceeds
1, the largest value the gap can take.

1. For one fixed $h$, the training error is an average of $n$ independent 0/1 variables with mean $R(h)$, and Hoeffding's inequality says that such an average is unlikely to be far from its mean: $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$. Derive the finite-class bound stated above from this and the fact that the probability of a union of events is at most the sum of their probabilities.
2. A network with $W=10^6$ float32 weights is a finite class, since each weight takes at most $2^{32}$ values, so $|\mathcal H|\le2^{32W}$. Evaluate the bound for $n=5\times10^4$ (the size of the CIFAR-10 training set) and $\delta=0.05$. Then compare $n$ with $d\approx W\ln W$, a general upper bound on the *VC dimension* of ReLU networks of fixed depth. §3 defines the VC dimension; for now, take it as the largest number of inputs the class can label in every possible way, and take as given that a VC bound on the gap is small only when $n\gg d$.
3. Zhang et al. show that this kind of architecture, trained by SGD, fits CIFAR-10 images with uniformly random labels to zero training error. Let $\mathcal H$ be the set of functions the architecture and SGD can produce, and write $\mathrm{gap}(h)=R(h)-\hat R_n(h)$. Assume that this fitting succeeds with high probability on i.i.d. samples from a distribution whose labels are uniform and independent of the image, and whose inputs are distinct. Show that every bound of the form "for every data distribution $P$, with probability $\ge1-\delta$, all $h\in\mathcal H$ have $\mathrm{gap}(h)\le\epsilon(n,\delta)$" must have $\epsilon\ge0.9$ (10 classes), up to sampling fluctuation.

<details><summary>Hint</summary>

For 3, the bound has to hold for *every* $P$. Choose one.
</details>
"""
    ),
    answer(
        r"""
**1.** For a fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$. The event that *some* $h\in\mathcal H$ has a gap larger than $t$ is the union over $h$ of these events, so its probability is at most $2|\mathcal H|e^{-2nt^2}$. Setting this equal to $\delta$ and solving for $t$ gives $t=\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.

**2.** $\ln|\mathcal H|\le32W\ln2\approx2.22\times10^7$, so the bound is $\sqrt{(2.22\times10^7+\ln 40)/10^5}\approx\sqrt{222}\approx14.9$. The gap cannot exceed 1, so a bound of 14.9 says nothing. The VC scale $W\ln W\approx1.4\times10^7$ is far larger than $n=5\times10^4$, so a VC bound is vacuous too. (This scale is a generic upper bound; it does not prove that this architecture has so large a VC dimension. But if the true VC dimension is at least $n$, then the class can label some set of $n$ inputs in every possible way, and the no-free-lunch argument of 3.1.1 applies: for inputs drawn uniformly from that set, a training sample of size $n$ misses an expected fraction $(1-1/n)^n\approx e^{-1}$ of them, and some labelling of the missed inputs makes any learner wrong on half of them, so the expected error is at least $\tfrac12(1-1/n)^n\approx0.18$.)

**3.** Take $P'$ to be CIFAR-10 images with labels drawn uniformly from the 10 classes, independently of the image. By assumption SGD then reaches an $h\in\mathcal H$ with $\hat R_n(h)=0$. The label of a fresh example is independent of its image, so any classifier is right with probability $1/10$: $R(h)=0.9$, and $\mathrm{gap}(h)=0.9$ with high probability. A bound that depends only on $\mathcal H$, $n$ and $\delta$ must hold for $P'$ in particular, so $\epsilon(n,\delta)\ge0.9-o(1)$. The same bound then applies, and is vacuous, on the real labels as well. (Zhang et al.'s experiment is on one fixed image collection; it motivates the assumption that fitting succeeds on random draws, but does not prove it.) A vacuous bound is not wrong, but it cannot be the explanation. 1.2.1 asks which explanations survive.
"""
    ),
    markdown(
        r"""
**1.2. Questions on Zhang et al.**

1. List the explanations of generalization that the random-labels experiment rules out, and those it leaves standing.
2. The paper finds that the usual explicit regularizers help but are neither necessary nor sufficient for generalization. These are weight decay, which adds a penalty $\lambda\|\theta\|^2$ on the weights $\theta$ to the training loss; dropout, which randomly zeroes units during training; and data augmentation, which trains on transformed copies of the images. The weight-decay penalty has three standard readings, each derived in §4. It is a Gaussian *prior* on the weights, so that minimizing the penalized loss finds the most probable weights given the data. It is a *constraint* that keeps the weights inside a ball of fixed radius, a smaller class to which classical bounds might apply. And it is a *spectral filter*: it shrinks the components of the weights along the directions that the data barely determine, which gradient descent stopped early also does. Which reading does the finding count against, and which survive?
"""
    ),
    answer(
        r"""
**1.** *Ruled out:*
- any explanation in which the class of reachable functions is small enough for a bound that holds for every distribution to be tight. This covers the VC dimension, and also the Rademacher complexity, which measures how well functions in the class can fit random $\pm1$ labels on the training inputs: fitting random labels pushes it to its maximum, as the paper notes in §2.2;
- explicit regularization as the main cause, since the networks generalize nearly as well without it;
- "SGD cannot find memorizing solutions", since it finds them, only a constant factor more slowly.

*Not ruled out:*
- data-dependent accounts, in which what is learned depends on structure in the true labels: a preference of the training procedure for low-norm, flat or simple solutions *when such solutions exist*, an *implicit bias* of training;
- bounds evaluated on the particular function found rather than on the whole class, in terms of its weight norms, its classification margin, how compressible it is, or its probability under a prior (PAC-Bayes, §6);
- a prior over functions whose support is everything but whose mass is on structured functions.

This is what the random-labels experiment alone leaves standing; §6 and §8 constrain the second item further.

The slower fit on random labels is a hint that some simplicity ordering is at work. Such a "soft" bias (the class can fit anything, but training prefers some fits) predicts things a hard restriction on the class does not:
- the same model at the same $n$ generalizes differently depending on the labels;
- random labels take longer to fit and end at larger norm or complexity;
- adding parameters need not hurt, and can help (double descent, 1.3–1.6), whereas a bound based on the size of the class only gets worse as the class grows;
- a bound in terms of the prior mass or description length of the *found* function can be non-vacuous where class-size bounds are not.

**2.** It counts against the *constraint* reading used as an explanation: the networks generalize without the constraint, and with it they still fit random labels, so the constrained class is still far too large for a class-size bound. The *prior* reading survives if the prior is soft and implemented implicitly, by architecture, initialization and SGD. The *spectral-filter* reading also survives as implicit regularization: gradient descent fits the directions the data determine strongly first, stopping early leaves the others unfit, and from zero initialization gradient descent converges to the smallest-norm fit (4.3).
"""
    ),
    markdown(
        r"""
<a id="wb2-double-descent"></a>

### Exercises: double descent in least squares

This exercise derives observation 2 above exactly. The textbook picture is
that test error follows a U, falling as model size grows, then rising once the
model has capacity to fit noise. In
least-squares regression with $p$ parameters and $n$ examples the rise is
real, and it peaks at the interpolation threshold $p=n$, where the
minimum-norm interpolant inverts near-zero singular values and its variance
diverges. Past the threshold the test error falls again. A well-tuned ridge
penalty removes the peak, and for signal-to-noise ratio above one the
interpolant eventually beats even the null predictor $\hat y=0$. §5 asks when
interpolation costs almost nothing. Sources: [Hastie, Montanari, Rosset and
Tibshirani (2022)](https://arxiv.org/abs/1903.08560) for the curves (reading
in §5), [Nakkiran, Venkat, Kakade and Ma (2020)](https://arxiv.org/abs/2003.01897)
for 1.5.3.

**Setup for 1.3–1.6.** The data follow a linear model $y=x^\top\beta+\varepsilon$
with $x,\beta\in\mathbb R^p$: $\beta$ is the true coefficient vector and
$\varepsilon$ is noise. There are $n$ training examples. Stacking their inputs
as rows gives the $n\times p$ matrix $X$ (the *design matrix*) and their
outputs the vector $y\in\mathbb R^n$, so $y=X\beta+\varepsilon$. Take the
inputs $x_i\sim\mathcal N(0,I_p)$ independently, the noise
$\varepsilon\sim\mathcal N(0,\sigma^2I_n)$ independent of $X$, and
$\|\beta\|^2=r^2$ ($r,\sigma>0$); $r^2/\sigma^2$ is the *signal-to-noise
ratio*. Write $\gamma=p/n$ for the number of parameters per example. The
*risk* of an estimate $\hat\beta$ is its expected squared prediction error on
a fresh input $x_0\sim\mathcal N(0,I_p)$, beyond the unavoidable noise
$\sigma^2$, averaged over the training data:
$\mathcal R=\mathbb E\big[(x_0^\top\hat\beta-x_0^\top\beta)^2\big]=\mathbb E\|\hat\beta-\beta\|^2$,
the second equality because $\mathbb E[x_0x_0^\top]=I$. Predicting zero
($\hat\beta=0$) has risk $r^2$. *Least squares* chooses $\hat\beta$ to
minimize $\|y-X\hat\beta\|^2$. *Ridge regression* adds a penalty on the size
of the coefficients,
$\hat\beta_\lambda=\arg\min_b\big[\|y-Xb\|^2+n\lambda\|b\|^2\big]=(X^\top X+n\lambda I)^{-1}X^\top y$.
Limits are $p,n\to\infty$ with $p/n\to\gamma$. One fact is quoted (derived in
2.3): for $p<n-1$, $\mathbb E[(X^\top X)^{-1}]=I_p/(n-p-1)$, so least squares
has $\mathcal R=\sigma^2p/(n-p-1)$.

**1.3. The left branch: fewer parameters than examples.** For $p<n$ the
columns of $X$ are linearly independent with probability one, so $X^\top X$
is invertible and the least-squares estimator
$\hat\beta=(X^\top X)^{-1}X^\top y$ is unique.

1. From $\mathcal R=\sigma^2p/(n-p-1)$, derive the limit $\sigma^2\gamma/(1-\gamma)$, which diverges as $\gamma\to1$. Show that $\mathcal R=\sigma^2\,\mathbb E[\mathrm{Tr}(X^\top X)^{-1}]=\frac{\sigma^2}{n}\mathbb E\big[\sum_i1/\lambda_i\big]$, where $\lambda_i$ are the eigenvalues of $X^\top X/n$, and use this to say which directions in parameter space are responsible for the divergence. You may quote the [Marchenko–Pastur law](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution): for $p,n\to\infty$ with $p/n\to\gamma\le1$, the eigenvalues of $X^\top X/n$ fill the interval $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$. It holds for any $X$ with i.i.d. entries of mean zero and unit variance, Gaussian or not.
2. The *bias* of an estimator is $\mathbb E[\hat\beta]-\beta$. What is the bias of least squares for $\gamma<1$?
"""
    ),
    answer(
        r"""
**1.** $\sigma^2\frac{p}{n-p-1}=\sigma^2\frac{p/n}{1-p/n-1/n}\to\sigma^2\frac{\gamma}{1-\gamma}$. For the trace formula, $\hat\beta-\beta=(X^\top X)^{-1}X^\top\varepsilon$, so at fixed $X$, $\mathbb E[\|\hat\beta-\beta\|^2\mid X]=\sigma^2\mathrm{Tr}\big[(X^\top X)^{-1}X^\top X(X^\top X)^{-1}\big]=\sigma^2\mathrm{Tr}(X^\top X)^{-1}$, and $\mathrm{Tr}(X^\top X)^{-1}=\frac1n\sum_i1/\lambda_i$. By the Marchenko–Pastur law the eigenvalues fill $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$, and as $\gamma\to1$ the lower edge tends to $0$, so the smallest eigenvalues, the directions in which the data barely determine $\beta$, blow up the sum. Consistently, $\frac1p\mathrm{Tr}(X^\top X/n)^{-1}\to\frac1{1-\gamma}$.

**2.** Zero: $\mathbb E[\hat\beta\mid X]=(X^\top X)^{-1}X^\top X\beta=\beta$ whenever $X^\top X$ is invertible (2.2). Below the threshold the whole risk is variance, the fluctuation of $\hat\beta$ around $\beta$ caused by the noise.
"""
    ),
    markdown(
        r"""
**1.4. The right branch: more parameters than examples.** For $p>n$ the rows
of $X$ are linearly independent with probability one, so $XX^\top$ is
invertible, and infinitely many coefficient vectors $b$ fit the training data
exactly. Among these, gradient descent from $b=0$
reaches the one of smallest norm (4.3.4), which is also the limit of the
ridge estimator $\hat\beta_\lambda$ as $\lambda\to0^+$; call it the
*minimum-norm interpolant*. Write
$X^+=X^\top(XX^\top)^{-1}$ (the Moore–Penrose pseudoinverse of $X$).

1. Show that $\hat\beta=X^+y$ interpolates, and that it has the smallest norm of all interpolants.
2. Let $P=X^+X$. Show that $P$ is the orthogonal projection onto the row space of $X$, the span of the training inputs, and that $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$ with the two terms orthogonal. So $\mathcal R=\underbrace{\mathbb E[\|(I-P)\beta\|^2]}_{\text{bias}^2}+\underbrace{\sigma^2\,\mathbb E[\mathrm{Tr}(XX^\top)^{-1}]}_{\text{variance}}$: the first term is the part of $\beta$ that the training inputs never probe, the second is the noise the fit absorbs.
3. *Bias.* Show that $\mathbb E[P]=(n/p)I$, so that $\text{bias}^2=r^2(1-n/p)\to r^2(1-1/\gamma)$.
4. *Variance.* $XX^\top=\sum_{j=1}^px_{\cdot j}x_{\cdot j}^\top$ is a sum of $p$ outer products of independent $\mathcal N(0,I_n)$ vectors, the columns of $X$, which is exactly the form $X^\top X$ had with the roles of $n$ and $p$ exchanged. Reuse the quoted fact to get variance $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$.
5. Assemble the full curve for $\gamma\in(0,\infty)$. Describe it near $\gamma=1$ and as $\gamma\to\infty$.

<details><summary>Hint</summary>

For 3, use rotation invariance: for any orthogonal $O\in\mathbb R^{p\times p}$, $XO$ has the same distribution as $X$. Combine this with $\mathrm{Tr}\,P=n$.
</details>
"""
    ),
    answer(
        r"""
**1.** $X\hat\beta=XX^\top(XX^\top)^{-1}y=y$, so $\hat\beta$ interpolates. Any other interpolant is $\hat\beta+w$ with $Xw=0$. Now $\hat\beta=X^\top(XX^\top)^{-1}y$ is a linear combination of the rows of $X$, while $Xw=0$ says $w$ is orthogonal to every row, so $\hat\beta\perp w$ and $\|\hat\beta+w\|^2=\|\hat\beta\|^2+\|w\|^2\ge\|\hat\beta\|^2$.

**2.** $P=X^\top(XX^\top)^{-1}X$ is symmetric and satisfies $P^2=P$, so it is an orthogonal projection; its range lies in the row space (it is $X^\top$ times something) and $PX^\top=X^\top$, so it fixes the row space and is the projection onto it. Then $\hat\beta=X^+(X\beta+\varepsilon)=P\beta+X^+\varepsilon$, so $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$. The first term is orthogonal to the row space and the second lies in it, so the cross term vanishes. Here "bias squared" means the squared bias at fixed $X$, then averaged over $X$, rather than the bias after averaging over $X$. Finally $\mathbb E[\|X^+\varepsilon\|^2]=\sigma^2\mathrm{Tr}(X^{+\top}X^+)$ and $X^{+\top}X^+=(XX^\top)^{-1}XX^\top(XX^\top)^{-1}=(XX^\top)^{-1}$.

**3.** For orthogonal $O$, $XO$ has the same distribution as $X$ (each row is a rotated standard Gaussian vector, which is again standard Gaussian), and its row-space projection is $O^\top PO$. So $\mathbb E[P]=O^\top(\mathbb E[P])O$ for all $O$, which forces $\mathbb E[P]=cI$. Taking traces, $\mathrm{Tr}\,P=n$ (a projection onto an $n$-dimensional space) gives $c=n/p$. For any fixed $\beta$, $\mathbb E[\|(I-P)\beta\|^2]=\beta^\top(I-\mathbb E[P])\beta=r^2(1-n/p)$. The interpolant only sees the component of $\beta$ along the training inputs, a random $n$-dimensional subspace, and misses the rest.

**4.** For $p>n+1$, the quoted fact with $(n,p)\to(p,n)$ gives $\mathbb E[(XX^\top)^{-1}]=I_n/(p-n-1)$, so the variance is $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$. (Hastie et al. state this branch as part of their Theorem 1. For independent Gaussian inputs it is a two-line consequence of the same fact.)

**5.**

$$
\mathcal R(\gamma)=\begin{cases}\sigma^2\dfrac{\gamma}{1-\gamma},&\gamma<1,\\[2mm] r^2\Big(1-\dfrac1\gamma\Big)+\dfrac{\sigma^2}{\gamma-1},&\gamma>1.\end{cases}
$$

The risk starts at $0$, rises, and diverges at $\gamma=1$ from both sides; the divergence is entirely variance. Past the threshold, the bias grows from $0$ toward $r^2$ while the variance falls from $\infty$ toward $0$. As $\gamma\to\infty$ the risk tends to $r^2$, the risk of predicting zero: the training inputs span a vanishing fraction of the directions of $\beta$, and the noise is spread over so many directions that it does no harm. The "second descent" is the variance falling faster than the bias rises.
"""
    ),
    markdown(
        r"""
**1.5. When does interpolation beat doing nothing, and what removes the
peak?** Two facts are used. First, suppose one believes before seeing data
that $\beta\sim\mathcal N(0,\tau^2I)$. Then the posterior mean of $\beta$
given the data, which is the estimate that minimizes the squared error
averaged over this prior and the noise, is exactly the ridge estimator
$\hat\beta_\lambda$ with $\lambda=\sigma^2/(n\tau^2)$ (4.1.1). Second, the
minimum-norm interpolant is the limit of $\hat\beta_\lambda$ as
$\lambda\to0^+$. Write $\lambda^*=\sigma^2\gamma/r^2$.

1. For $\gamma>1$, minimize the risk of the minimum-norm interpolant, $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$ over $\gamma$. Show that if $r>\sigma$ (signal-to-noise ratio above one), the minimum is at $\gamma=r/(r-\sigma)$, with value $2r\sigma-\sigma^2<r^2$. Show that if $r\le\sigma$, the risk decreases monotonically toward $r^2$ from above.
2. Show that $\lambda^*$ equals $\sigma^2/(n\tau^2)$ for the prior $\beta\sim\mathcal N(0,(r^2/p)I)$, which has $\mathbb E[\|\beta\|^2]=r^2$. Hence $\hat\beta_{\lambda^*}$ is the best possible estimator in squared error averaged over that prior.
3. Use 2 to argue that (a) the risk of $\hat\beta_{\lambda^*}$ is at most the risk of the minimum-norm interpolant at every $\gamma$, and (b) at fixed $p$ the risk of $\hat\beta_{\lambda^*}$ is non-increasing in $n$. Deduce that its limiting curve is non-decreasing in $\gamma$, so it has no peak.

<details><summary>Hint</summary>

For 3(a), rotating the inputs and the coefficients together ($X\to XO$, $\beta\to O^\top\beta$) leaves both estimators' predictions unchanged, so each one's risk depends on $\beta$ only through $\|\beta\|$. Both estimators are also linear in $y$, so at fixed $\beta$ the risk is a quadratic form in $\beta$ plus a constant; with the symmetry this makes it $A\|\beta\|^2+B$, and averaging over the prior then gives the same value as any fixed $\beta$ of norm $r$. For 3(b), what can an estimator with $n+1$ samples always do to guarantee it is no worse than one with $n$?
</details>
"""
    ),
    answer(
        r"""
**1.** $\frac{d}{d\gamma}\big[r^2(1-1/\gamma)+\sigma^2/(\gamma-1)\big]=\frac{r^2}{\gamma^2}-\frac{\sigma^2}{(\gamma-1)^2}$. This vanishes when $\frac{\gamma-1}{\gamma}=\frac\sigma r$, i.e. $\gamma^\star=\frac{1}{1-\sigma/r}=\frac{r}{r-\sigma}$, which requires $r>\sigma$. There $1-1/\gamma^\star=\sigma/r$ and $\gamma^\star-1=\sigma/(r-\sigma)$, so

$$
\mathcal R(\gamma^\star)=r\sigma+\sigma(r-\sigma)=2r\sigma-\sigma^2=r^2-(r-\sigma)^2<r^2 .
$$

The derivative is negative before $\gamma^\star$ and positive after, and $\mathcal R\to r^2$ from below as $\gamma\to\infty$. So for signal-to-noise ratio above one, the interpolant beats predicting zero over a whole range of $\gamma$. If $r\le\sigma$, then $\frac{\gamma-1}{\gamma}<1\le\frac\sigma r$, the derivative is negative everywhere on $(1,\infty)$, and the risk falls toward $r^2$ without crossing it: the interpolant never beats predicting zero.

In this model the global minimum over all $\gamma$ is still at $\gamma\to0$, because $r^2$ is held fixed as $p$ grows, so each added feature carries less signal. Whether adding features can win globally depends on whether they carry new signal. Hastie et al. (§5) study a variant in which the true function depends on features the model leaves out, so that adding features brings in signal the model was missing; there, for signal-to-noise ratio above one, the global minimum can lie at $\gamma>1$.

**2.** With $\tau^2=r^2/p$: $\lambda=\frac{\sigma^2}{n\tau^2}=\frac{\sigma^2p}{nr^2}=\frac{\sigma^2\gamma}{r^2}=\lambda^*$. By 4.1.1, $\hat\beta_{\lambda^*}$ is the posterior mean, which minimizes $\mathbb E[\|b-\beta\|^2]$ over all estimators $b$ when $\beta$ is drawn from that prior. (This is also 2.5.3's best common penalty $n\lambda=\sigma^2/\tau^2$.)

**3.** (a) Both estimators are unchanged by rotating inputs and coefficients together, and the distribution of $X$ is rotation-invariant, so each one's risk at fixed $\beta$ depends only on $\|\beta\|$. More precisely, for either estimator the risk at fixed $\beta$ has the form $A\|\beta\|^2+B$: by the same symmetry the squared-bias matrix is a multiple of $I$, and the noise term does not involve $\beta$. So the risk averaged over any prior with $\mathbb E[\|\beta\|^2]=r^2$ equals the risk at any single vector of norm $r$, already at finite $p$ (whenever the risks are finite). $\hat\beta_{\lambda^*}$ has the smallest prior-averaged risk of any estimator, so it is at least as good as the minimum-norm interpolant at every fixed $\beta$ of norm $r$. (This compares the two estimators at fixed norm; it does not make $\hat\beta_{\lambda^*}$ the best of all estimators for a $\beta$ drawn uniformly from the sphere.)

(b) An estimator given $n+1$ samples can ignore the last one, so the best estimator with $n+1$ samples has prior-averaged risk at most that of the best estimator with $n$ samples. More data never hurts the posterior mean. The limiting curve depends on $\gamma$ alone, and at fixed $p$ increasing $n$ decreases $\gamma$, so the risk of $\hat\beta_{\lambda^*}$ is non-decreasing in $\gamma$. There is no peak.

The double-descent peak is therefore a property of the minimum-norm estimator, a variance divergence at $p\approx n$ that properly tuned regularization removes, and not a property of the problem.
"""
    ),
    markdown(
        r"""
**1.6. The curves at two signal-to-noise ratios.** For $r^2/\sigma^2=5$ and
for $r^2/\sigma^2=0.5$, use 1.5.1 to locate the minimum of the risk of the minimum-norm interpolant
on $\gamma>1$, if it has one, and say whether the curve crosses $r^2$, the
risk of predicting zero. Then sketch both curves against $\gamma$ on a log
axis, with $r^2$ as a horizontal line, the bias and variance contributions
shaded, and the risk of $\hat\beta_{\lambda^*}$ on the same axes, using what 1.5.3 says
about its shape. The interactive figure below draws all of these.
"""
    ),
    answer(
        r"""
*What a correct sketch shows.* For both signal-to-noise ratios, the curve starts at $0$, rises as $\sigma^2\gamma/(1-\gamma)$, and has a vertical asymptote at $\gamma=1$, entirely from variance; the bias is zero for $\gamma<1$.
- For $r^2/\sigma^2=5$ ($r\approx2.24\sigma$): past the peak the curve falls *through* $r^2$, has its minimum at $\gamma=r/(r-\sigma)\approx1.81$ with value $2r\sigma-\sigma^2\approx3.47\sigma^2<r^2=5\sigma^2$, and then climbs back to $r^2$ from below.
- For $r^2/\sigma^2=0.5$: past the peak the curve falls toward $r^2$ from above and never crosses it.

The risk of $\hat\beta_{\lambda^*}$ is a smooth, non-decreasing curve that lies below that of the minimum-norm interpolant everywhere and tends to $r^2$ as $\gamma\to\infty$.
"""
    ),
    markdown(
        figure(
            "assets/interactive/double_descent.html",
            "Turn on the bias/variance decomposition and check which term makes the peak. Toggle optimal "
            "ridge λ* = σ²γ/r². Drag the SNR across 1 and watch whether the min-norm curve dips below "
            "the null risk r². The dots are finite-n simulations: note how they scatter near γ = 1.",
        )
    ),
    markdown(
        r"""
**1.7. Is a network's peak a variance divergence?** Networks show
double-descent curves in width, training time and sample size (Nakkiran et al.
2019), usually with label noise, and regularization or early stopping weakens
them. The expected squared test error of a learned predictor splits into three
parts (2.4.1): the noise, the squared bias of the predictor averaged over
training sets, and the variance of the predictor across training sets.
Describe a measurement that would tell you whether a network's double-descent
peak is a variance divergence of the kind derived in 1.4.
"""
    ),
    answer(
        r"""
Train several networks of the same size on independent draws of the training data (or with different seeds and different label noise) near the apparent peak. Split the test loss into the loss of the averaged prediction (bias plus noise) and the spread of the individual predictions around it (variance), as in 2.4. If the peak is a variance divergence, the spread peaks there while the loss of the averaged prediction does not. Weight decay tuned separately at each size should then flatten the peak. A second check is the eigenvalues of whatever matrix plays the role of $X^\top X/n$ for the network, its feature covariance: a variance divergence of the kind in 1.3 requires eigenvalues near zero.
"""
    ),
]

CELLS = OPENING + KEYSTONE
