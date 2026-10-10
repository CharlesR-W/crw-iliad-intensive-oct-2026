"""WB2 opening and §1 keystone: vacuous bounds and double descent.

Exercises 1.3-1.6 are checked by checks/wb2a_check.py (old numbering 3.A-3.C).
"""
from nbkit import *  # noqa: F401,F403

OPENING = [
    markdown(
        r"""
# Workbook 2 — How Deep Learning Changed Statistical Learning Theory

**Mysteries of Deep Learning, Iliad Intensive 2026**

Classical learning theory says that a model able to fit anything, which then
fits its training data exactly, has no reason to generalize. Deep networks do
both and generalize anyway. This workbook is about what that broke and what
survives; it is the derivations workbook, so most sections end in exercises
with worked solutions folded beneath them.

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

These two observations broke the textbook account of generalization; every
later section of this workbook is a response to one of them.
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

**Bounds that hold for every distribution cannot explain this.** A classical
generalization bound depends only on the model class, the sample size $n$ and
a confidence level. Zhang et al. trained standard CIFAR-10 networks on labels
drawn at random, and the networks still reached zero training error. So the
functions training can reach include ones with training error 0 and test
error 0.9, and any such bound must allow a gap of 0.9, also on the real labels,
where the same networks generalize well. Nagarajan and Kolter went further:
in their examples even uniform convergence over just the functions gradient
descent returns is vacuous. An explanation has to use the data and the
training procedure, not only the class.

**Fitting the training data exactly is not overfitting.** The textbook picture
is a U: test error falls as the model grows, then rises once it starts fitting
noise. Past the interpolation threshold, where the number of parameters $p$
reaches the number of examples $n$, test error falls again: double descent.
Least-squares regression already shows it. The peak at $p\approx n$ is a
variance divergence of the minimum-norm interpolant, well-tuned ridge removes
it, and past it the interpolant can beat predicting zero. The exercises below
derive this exactly; §5 asks when interpolation costs almost nothing.

### Exercises: capacity bounds meet a real network

These make the first idea quantitative.

**1.1. Plug in a real network.** Use the 0–1 loss, so the population risk
$R(h)$ of a classifier $h$ is its test error and the empirical risk
$\hat R_n(h)$ its training error on $n$ i.i.d. examples. For a finite class
$\mathcal H$ the classical uniform bound is: with probability at least
$1-\delta$, $\sup_{h\in\mathcal H}|R(h)-\hat R_n(h)|\le\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.
A bound is *vacuous* if it exceeds 1, the largest possible loss.

1. Derive the finite-class bound from Hoeffding's inequality for one fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$, and a union bound.
2. A network with $W=10^6$ float32 weights is a finite class: each weight takes at most $2^{32}$ values. Evaluate the bound for $n=5\times10^4$ (the size of the CIFAR-10 training set) and $\delta=0.05$. Then compare $n$ with the VC-dimension scale $d\approx W\ln W$ (the general upper bound for ReLU networks of fixed depth; §3 defines VC dimension). A VC bound needs $n\gg d$.
3. Zhang et al. show that this kind of architecture, trained by SGD, fits CIFAR-10 images with uniformly random labels to zero training error. Let $\mathcal H$ be the set of functions the architecture and SGD can produce, and write $\mathrm{gap}(h)=R(h)-\hat R_n(h)$. Assume such interpolation succeeds with high probability on i.i.d. samples from a distribution with independent uniform labels and distinct inputs. Show that every bound of the form "for every data distribution $P$, with probability $\ge1-\delta$, all $h\in\mathcal H$ have $\mathrm{gap}(h)\le\epsilon(n,\delta)$" must have $\epsilon\ge0.9$ (10 classes), up to sampling fluctuation.

<details><summary>Hint</summary>

For 3, the bound has to hold for *every* $P$. Choose one.
</details>
"""
    ),
    answer(
        r"""
**1.** For a fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$. A union bound over $|\mathcal H|$ hypotheses gives $\Pr[\sup_h|\cdot|>t]\le2|\mathcal H|e^{-2nt^2}$. Setting this equal to $\delta$ and solving gives $t=\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.

**2.** Using the parameter-encoding upper bound $\ln|\mathcal H|\le32W\ln2\approx2.22\times10^7$, so the bound is $\sqrt{(2.22\times10^7+\ln 40)/10^5}\approx\sqrt{222}\approx14.9$. Since the 0–1 loss is at most 1, this parameter-encoding bound says nothing. The rough VC scale is $W\ln W\approx1.4\times10^7\gg n=5\times10^4$, and the VC bound needs $n\gg d$, so it is vacuous too. This scale is a generic upper bound; it does not prove that this architecture has that VC dimension. But if the actual VC dimension is at least $n$, NFL bites as well: the class shatters some $n$ points, and for $D$ uniform on those points the sample misses an expected fraction $(1-1/n)^n\approx e^{-1}$ of them, so by 3.1.1 some target forces expected error at least $\tfrac12(1-1/n)^n\approx0.18$ on any learner.

**3.** Take $P'$ to be CIFAR-10 images with labels drawn uniformly from the 10 classes, independently of the image. Assume the random-label fitting result extends to i.i.d. samples from this input distribution, with distinct inputs and high probability of interpolation. Then SGD reaches an $h\in\mathcal H$ with $\hat R_n(h)=0$. Its labels are independent of the input, so $R(h)=0.9$. So on $P'$, some $h\in\mathcal H$ has gap $0.9$ with high probability. A bound that depends only on $\mathcal H$, $n$ and $\delta$ must hold for $P'$, so $\epsilon(n,\delta)\ge0.9-o(1)$. The same distribution-independent bound is then vacuous for the real labels as well. The experiment on one fixed image collection motivates the interpolation assumption; it does not prove it for every random draw. Vacuous is not wrong, but it cannot be the explanation. 1.2.1 asks which explanations survive.
"""
    ),
    markdown(
        r"""
**1.2. Questions on Zhang et al.**

1. List the explanations of generalization that the random-labels experiment rules out, and those it leaves standing.
2. The paper finds that explicit regularizers (weight decay, dropout, data augmentation) help but are neither necessary nor sufficient. An $L^2$ penalty has three standard readings (§4 derives each): a Gaussian *prior* on the weights; a *constraint* to a norm ball whose small capacity guarantees generalization; and a *spectral filter* that suppresses the directions the data barely determine, which gradient descent with early stopping also implements. Which reading does the finding count against, and which survive?
"""
    ),
    answer(
        r"""
**1.** *Ruled out:*
- any explanation through small capacity of the reachable class that holds for every distribution (VC dimension, Rademacher complexity of the class; as the paper notes in §2.2, fitting random labels means the empirical Rademacher complexity is essentially 1);
- explicit regularization as the main cause, since the nets generalize nearly as well without it;
- "SGD cannot find memorizing solutions", since it finds them, only a constant factor more slowly.

*Not ruled out:*
- data-dependent accounts, in which the learned function depends on structure in the true labels: an implicit bias toward low-norm, flat or simple solutions *when such solutions exist*;
- norm-, margin-, compression- and PAC-Bayes-style bounds evaluated on the learned function (§6);
- a prior over functions whose support is everything but whose mass is on structured functions.

The slower fit on random labels is a hint that some simplicity ordering is at work. Such a "soft" bias (the class can fit anything, but training prefers some fits) predicts things a hard capacity restriction does not:
- the same model at the same $n$ generalizes differently depending on the labels;
- random labels take longer to fit and end at larger norm or complexity;
- adding parameters need not hurt, and can help (double descent, 1.3–1.6), whereas a class-wide capacity upper bound can worsen with class size without predicting that actual risk must worsen;
- a bound in terms of the prior mass or description length of the *found* function can be non-vacuous where class-capacity bounds are not.

**2.** It counts against the *constraint* view used as an explanation, in which a norm ball is small enough for its capacity to explain generalization. The nets generalize without the constraint, and with the constraint they still fit random labels, so the constrained class is still huge. The *prior* view survives if the prior is soft and implemented implicitly, by architecture, initialization and SGD. The *spectral-filter* view also survives as implicit regularization: early stopping and gradient descent's preference for large-eigenvalue directions, and gradient descent's convergence to the minimum-norm solution (4.3).
"""
    ),
    markdown(
        r"""
<a id="wb2-double-descent"></a>

### Exercises: double descent in least squares

The second idea, derived exactly. Sources: [Hastie, Montanari, Rosset and
Tibshirani (2022)](https://arxiv.org/abs/1903.08560) for the curves (reading
in §5), [Nakkiran, Venkat, Kakade and Ma (2020)](https://arxiv.org/abs/2003.01897)
for 1.5.3.

**Setup for 1.3–1.6.** The model is $y=x^\top\beta+\varepsilon$ with
$x,\beta\in\mathbb R^p$. There are $n$ training examples, the rows of the
design matrix $X\in\mathbb R^{n\times p}$, with rows $x_i\sim\mathcal N(0,I_p)$,
noise $\varepsilon\sim\mathcal N(0,\sigma^2)$, $\|\beta\|^2=r^2$ ($r,\sigma>0$),
and $\gamma=p/n$. The *risk* of an estimate $\hat\beta$ is its excess squared
error on a fresh test input $x_0\sim\mathcal N(0,I_p)$, averaged over the
training data: $\mathcal R=\mathbb E\big[(x_0^\top\hat\beta-x_0^\top\beta)^2\big]=\mathbb E\|\hat\beta-\beta\|^2$.
It excludes the irreducible $\sigma^2$. Predicting zero has risk $r^2$. Ridge
is $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$. Limits are
$p,n\to\infty$ with $p/n\to\gamma$. One fact is quoted (derived in 2.3):
for $p<n-1$, $\mathbb E(X^\top X)^{-1}=I_p/(n-p-1)$, so least squares has
$\mathcal R=\sigma^2p/(n-p-1)$.

**1.3. The left branch.** For $p<n$ the least-squares estimator
$\hat\beta=(X^\top X)^{-1}X^\top y$ is unique.

1. From $\mathcal R=\sigma^2p/(n-p-1)$, derive the limit $\sigma^2\gamma/(1-\gamma)$. Explain the divergence in terms of the spectrum of $X^\top X/n$.
2. What is the bias of least squares for $\gamma<1$?
"""
    ),
    answer(
        r"""
**1.** $\sigma^2\frac{p}{n-p-1}=\sigma^2\frac{p/n}{1-p/n-1/n}\to\sigma^2\frac{\gamma}{1-\gamma}$. The empirical spectrum of $X^\top X/n$ converges to the Marchenko–Pastur law on $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (quoted). The risk is $\frac{\sigma^2}{n}\sum_i1/\lambda_i(X^\top X/n)$, and as $\gamma\to1$ the lower edge tends to $0$, so the smallest eigenvalues blow up the sum. Consistently, $\frac1p\mathrm{Tr}(X^\top X/n)^{-1}\to\frac1{1-\gamma}$.

**2.** Zero: OLS is unbiased whenever $X$ has full column rank (2.2). Below the threshold the whole risk is variance.
"""
    ),
    markdown(
        r"""
**1.4. The right branch: minimum-norm interpolation.** For $p>n$, $X$ has
full row rank almost surely, and infinitely many $b$ interpolate the data
($Xb=y$). $X^+$ denotes the Moore–Penrose pseudoinverse.

1. Show that the minimum-norm least-squares solution is $\hat\beta=X^+y=X^\top(XX^\top)^{-1}y$ and that it interpolates. It is also the $\lambda\to0^+$ limit of ridge, and the limit of gradient descent from $0$ (4.3.4).
2. Let $P=X^+X$ be the orthogonal projection onto the row space of $X$. Show that $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$, with the two terms orthogonal. So $\mathcal R=\underbrace{\mathbb E\|(I-P)\beta\|^2}_{\text{conditional bias}^2}+\underbrace{\sigma^2\,\mathbb E\,\mathrm{Tr}(XX^\top)^{-1}}_{\text{variance}}$.
3. *Bias.* Show that $\mathbb E P=(n/p)I$, so that $\text{bias}^2=r^2(1-n/p)\to r^2(1-1/\gamma)$.
4. *Variance.* $XX^\top$ is an $n\times n$ Wishart matrix with $p$ degrees of freedom. Reuse the quoted fact with the roles of $n$ and $p$ swapped to get $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$.
5. Assemble the full curve for $\gamma\in(0,\infty)$. Describe it near $\gamma=1$ and as $\gamma\to\infty$.

<details><summary>Hint</summary>

For 3, use rotation invariance: $X\to XO$ for orthogonal $O\in\mathbb R^{p\times p}$, together with $\mathrm{Tr}\,P=n$. For 4, $XX^\top=\sum_{j=1}^px_{\cdot j}x_{\cdot j}^\top$, where the columns $x_{\cdot j}\in\mathbb R^n$ are i.i.d. $\mathcal N(0,I_n)$.
</details>
"""
    ),
    answer(
        r"""
**1.** $XX^\top$ is invertible, so $X\hat\beta=XX^\top(XX^\top)^{-1}y=y$, and $\hat\beta$ interpolates. Any other interpolant is $\hat\beta+w$ with $Xw=0$. Since $\hat\beta\in\mathrm{row}(X)\perp\ker X$, we have $\|\hat\beta+w\|^2=\|\hat\beta\|^2+\|w\|^2$, so $\hat\beta$ is the minimum-norm interpolant.

**2.** $\hat\beta=X^+(X\beta+\varepsilon)=P\beta+X^+\varepsilon$, so $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$. The cross term is identically zero: $(I-P)\beta$ lies in the null space and $X^+\varepsilon$ in the row space. Here "bias squared" means the squared bias conditional on $X$, then averaged over designs, rather than bias after averaging over $X$. Then $\mathbb E\|X^+\varepsilon\|^2=\sigma^2\mathrm{Tr}(X^{+\top}X^+)$, and $X^{+\top}X^+=(XX^\top)^{-1}XX^\top(XX^\top)^{-1}=(XX^\top)^{-1}$.

**3.** For orthogonal $O$, $XO$ has the law of $X$, and its row-space projection is $O^\top PO$. So $\mathbb EP=O^\top(\mathbb EP)O$ for all $O$, which forces $\mathbb EP=cI$. Taking traces, $\mathrm{Tr}\,P=n$ gives $c=n/p$. For any fixed $\beta$, $\mathbb E\|(I-P)\beta\|^2=\beta^\top(I-\mathbb EP)\beta=r^2(1-n/p)$. The interpolator sees only the $n$-dimensional projection of $\beta$ onto a random subspace, and it misses the rest.

**4.** For $p>n+1$, by 2.3.1 with $(n,p)\to(p,n)$: $\mathbb E(XX^\top)^{-1}=I_n/(p-n-1)$, so the variance is $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$. (Hastie et al. state this branch as part of their Theorem 1. In the isotropic Gaussian case it is a two-line consequence of the same Wishart fact.)

**5.**

$$
\mathcal R(\gamma)=\begin{cases}\sigma^2\dfrac{\gamma}{1-\gamma},&\gamma<1,\\[2mm] r^2\Big(1-\dfrac1\gamma\Big)+\dfrac{\sigma^2}{\gamma-1},&\gamma>1.\end{cases}
$$

The risk starts at $0$, rises, and diverges at $\gamma=1$ from both sides; the divergence is entirely variance. Past the threshold, the bias grows from $0$ toward $r^2$ while the variance falls from $\infty$ toward $0$. As $\gamma\to\infty$ the risk tends to $r^2$, the null risk: the projection captures a vanishing fraction of $\beta$, and the noise is spread over so many directions that it does no harm. The "second descent" is the variance falling faster than the bias rises.
"""
    ),
    markdown(
        r"""
**1.5. When does interpolation beat doing nothing, and what removes the
peak?** Two facts are used. Ridge at $\lambda=\sigma^2/(n\tau^2)$ is the
posterior mean of $\beta$ under the prior $\beta\sim\mathcal N(0,\tau^2I)$ with
Gaussian noise (4.1.1), and the posterior mean minimizes the squared error
averaged over the prior. The minimum-norm interpolant is the $\lambda\to0^+$
limit of ridge. Write $\lambda^*=\sigma^2\gamma/r^2$.

1. For $\gamma>1$, minimize $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$. Show that if $r>\sigma$ (signal-to-noise ratio $r^2/\sigma^2>1$), the minimum is at $\gamma=r/(r-\sigma)$, with value $2r\sigma-\sigma^2<r^2$. Show that if $r\le\sigma$, the risk decreases monotonically toward $r^2$ from above.
2. Show that $\lambda^*$ is $\sigma^2/(n\tau^2)$ for the prior $\beta\sim\mathcal N(0,(r^2/p)I)$, which has $\mathbb E\|\beta\|^2=r^2$. Hence ridge at $\lambda^*$ is the Bayes-optimal estimator under squared loss for that prior.
3. Use 2 to argue that (a) the optimal-ridge risk is at most the min-norm risk at every $\gamma$, and (b) at fixed $p$ the optimal-ridge risk is non-increasing in $n$. Deduce that the limiting optimal-ridge curve is non-decreasing in $\gamma$, so it has no peak.

<details><summary>Hint</summary>

For 3(a), both estimators are rotation-equivariant ($X\to XO$, $\beta\to O^\top\beta$), so each one's risk depends on $\beta$ only through $\|\beta\|$. For 3(b), what can a Bayes estimator with $n+1$ samples always do that guarantees it is no worse than with $n$?
</details>
"""
    ),
    answer(
        r"""
**1.** $\frac{d}{d\gamma}\big[r^2(1-1/\gamma)+\sigma^2/(\gamma-1)\big]=\frac{r^2}{\gamma^2}-\frac{\sigma^2}{(\gamma-1)^2}$. This vanishes when $\frac{\gamma-1}{\gamma}=\frac\sigma r$, i.e. $\gamma^\star=\frac{1}{1-\sigma/r}=\frac{r}{r-\sigma}$, which requires $r>\sigma$. There $1-1/\gamma^\star=\sigma/r$ and $\gamma^\star-1=\sigma/(r-\sigma)$, so

$$
\mathcal R(\gamma^\star)=r\sigma+\sigma(r-\sigma)=2r\sigma-\sigma^2=r^2-(r-\sigma)^2<r^2 .
$$

The derivative is negative before $\gamma^\star$ and positive after, and $\mathcal R\to r^2$ from below as $\gamma\to\infty$. So for SNR above one, the interpolator beats the null estimator over a whole range of $\gamma$. If $r\le\sigma$, then $\frac{\gamma-1}{\gamma}<1\le\frac\sigma r$, the derivative is negative everywhere on $(1,\infty)$, and the risk falls toward $r^2$ without crossing it: the interpolator never beats predicting zero.

In this model the global minimum over all $\gamma$ is still at $\gamma\to0$, because $r^2$ is held fixed as $p$ grows, so each added feature carries less signal. Whether adding features can win globally depends on whether they carry new signal. Hastie et al. (§5) study a misspecified variant in which they do, and there, for SNR above one, the global minimum can lie at $\gamma>1$.

**2.** With $\tau^2=r^2/p$: $\lambda=\frac{\sigma^2}{n\tau^2}=\frac{\sigma^2p}{nr^2}=\frac{\sigma^2\gamma}{r^2}=\lambda^*$. By 4.1.1, ridge at $\lambda^*$ is the posterior mean, which minimizes $\mathbb E\|b-\beta\|^2$ over all estimators under that prior. (This is also 2.5.3's best common penalty $n\lambda=\sigma^2/\tau^2$.)

**3.** (a) Ridge and min-norm are both rotation-equivariant, and the design is rotation-invariant, so each one's risk at fixed $\beta$ depends only on $\|\beta\|$. Averaged over the prior, ridge($\lambda^*$) is no worse than any estimator. For either of these linear estimators, the noise- and design-averaged risk has the form $A\|\beta\|^2+B$: isotropy makes the squared-bias matrix a scalar multiple of $I$, and the noise variance is independent of $\beta$. Thus averaging over a prior with $\mathbb E\|\beta\|^2=r^2$ gives exactly its risk at any fixed vector of norm $r$, already at finite $p$ (when the risks are finite). The Bayes comparison therefore gives the fixed-radius comparison too. This does not make ridge Bayes-optimal under the uniform-on-a-sphere prior among all estimators.

(b) A Bayes estimator given $n+1$ samples can ignore the last one, so its Bayes risk is at most that of the Bayes estimator with $n$ samples. More data never hurts the posterior mean. The limiting curve depends on $\gamma$ alone, and at fixed $p$ increasing $n$ decreases $\gamma$, so the optimal-ridge risk is non-decreasing in $\gamma$. There is no peak.

The double-descent peak is therefore a property of the minimum-norm estimator, a variance divergence at $p\approx n$ that properly tuned regularization removes, and not a property of the problem.
"""
    ),
    markdown(
        r"""
**1.6. The curves at two signal-to-noise ratios.** For $r^2/\sigma^2=5$ and
for $r^2/\sigma^2=0.5$, use 1.5.1 to locate the min-norm risk's minimum on
$\gamma>1$, if it has one, and say whether the curve crosses the null risk
$r^2$. Then sketch both curves against $\gamma$ on a log axis, with $r^2$ as a
horizontal line, the bias and variance contributions shaded, and optimal ridge
on the same axes. The interactive figure below draws all of these.
"""
    ),
    answer(
        r"""
*What a correct sketch shows.* For both SNRs, the curve starts at $0$, rises as $\sigma^2\gamma/(1-\gamma)$, and has a vertical asymptote at $\gamma=1$, entirely from variance; the bias is zero for $\gamma<1$.
- For SNR $5$ ($r\approx2.24\sigma$): past the peak the curve falls *through* $r^2$, has its minimum at $\gamma=r/(r-\sigma)\approx1.81$ with value $2r\sigma-\sigma^2\approx3.47\sigma^2<r^2=5\sigma^2$, and then climbs back to $r^2$ from below.
- For SNR $0.5$: past the peak the curve falls toward $r^2$ from above and never crosses it.

Optimal ridge is a smooth, monotone, non-decreasing curve that lies below min-norm everywhere and tends to $r^2$ as $\gamma\to\infty$.
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
them. The expected squared test error of a learned predictor splits into
noise, squared bias of the average predictor, and variance across training
sets (2.4.1). Describe a measurement that would tell you whether a network's
double-descent peak is a variance divergence of the kind derived in 1.4.
"""
    ),
    answer(
        r"""
Train several networks at the same size on independent data draws (or with different seeds and different label noise) near the apparent peak. Decompose the test loss into the loss of the ensemble average (bias plus noise) and the spread across the ensemble (variance), as in 2.4. If the peak is a variance divergence, the spread peaks there while the ensemble-average loss does not. Weight decay tuned per size should then flatten the peak. A second check is the spectrum of the relevant kernel or feature covariance at the peak: there should be near-zero eigenvalues.
"""
    ),
]

CELLS = OPENING + KEYSTONE
