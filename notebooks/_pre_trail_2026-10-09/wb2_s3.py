"""WB2 §3: overfitting and double descent. Checked by checks/wb2a_check.py.
"""
from nbkit import *  # noqa: F401,F403
from wb2_new_benign import CELLS as S3_BENIGN  # noqa: F401  (3.D)

S3_ABC = [
    # ------------------------------------------------------------------ §3
    markdown(
        r"""
<a id="wb2-section-3"></a>

## 3. Overfitting and double descent

**The question.** Classical advice says a model with as many parameters as
data points will *overfit*: it fits the noise in the training set, and its
test error explodes. The textbook picture is a U-shaped curve of test error
against model size, with a sweet spot in the middle (1.C). Modern networks
have far more parameters than training examples, fit their training data
exactly (they *interpolate*), and generalize anyway. The curve that replaces
the U, "double descent", rises to a peak where the number of parameters $p$
matches the number of samples $n$, then falls again as $p$ grows further. The
name is from [Belkin et al. (2019)](https://arxiv.org/abs/1812.11118).

**What you get.** All of this already happens in least-squares regression,
where it can be computed exactly, and that is what this section does. In the
limit $p,n\to\infty$ with $p/n\to\gamma$:

- *Below the threshold* ($\gamma<1$, 3.A) least squares is unbiased, and its
  excess risk $\sigma^2\gamma/(1-\gamma)$ blows up as $\gamma\to1$.
- *Above it* ($\gamma>1$, 3.B) there are infinitely many interpolating
  solutions. The one gradient descent finds, the minimum-norm interpolant, has
  risk $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$. The peak at $\gamma=1$ is all
  variance.
- *Regularization removes the peak* (3.C): optimally tuned ridge has risk
  monotone in $\gamma$. And past the peak, when the signal-to-noise ratio
  exceeds one, the interpolator can beat predicting zero.
- *Benign overfitting* (3.D): when the extra dimensions are many and weak, the
  interpolator behaves like ridge, and fitting noise exactly costs almost
  nothing.
- *Which regime are we in?* (3.E) A short discussion of where LLMs sit.

The formulas are those of Hastie et al. (reading in 3.F).

**Conventions for 3.A–3.C (as in Hastie et al.).** The model is
$y=x^\top\beta+\varepsilon$ with $x,\beta\in\mathbb R^p$; there are $n$
training examples, stacked as rows of the design matrix $X\in\mathbb R^{n\times p}$,
and $\gamma=p/n$. Rows $x_i\sim\mathcal N(0,I_p)$, noise
$\varepsilon\sim\mathcal N(0,\sigma^2)$, and $\|\beta\|^2=r^2$, with $r,\sigma>0$. For an estimate
$\hat\beta$ the risk is its squared error on a fresh test input $x_0\sim\mathcal N(0,I_p)$,
$\mathcal R=\mathbb E\big[(x_0^\top\hat\beta-x_0^\top\beta)^2\big]=\mathbb E\|\hat\beta-\beta\|^2$,
averaged over the training data. It *excludes* the irreducible $\sigma^2$: it is the
expected excess risk $\mathbb E\,\mathcal E_n$ of 1.A, $\mathcal E_n=R(\hat\beta)-R(\beta)$,
under squared loss. Ridge is $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$.
Limits are $p,n\to\infty$ with $p/n\to\gamma$. The null estimator $\hat\beta=0$
("predict zero") has risk $r^2$; an estimator that does worse than $r^2$ is worse
than not learning at all.

### 3.A. Below the threshold

How does the risk of plain least squares grow as the number of parameters
approaches the number of samples from below, and why does it blow up?

#### 3.A.1. The left branch

For $p<n$ the least-squares estimator $\hat\beta=(X^\top X)^{-1}X^\top y$ is
unique. 1.B.2 computed its exact expected excess risk,
$\mathbb E\,\mathcal E_n=\sigma^2p/(n-p-1)$, using the Wishart identity
$\mathbb E(X^\top X)^{-1}=I_p/(n-p-1)$.

1. From 1.B.2 ($\mathbb E\,\mathcal E_n=\sigma^2p/(n-p-1)$), derive the limit $\sigma^2\gamma/(1-\gamma)$. Explain the divergence in terms of the spectrum of $X^\top X/n$.
2. What is the bias of OLS for $\gamma<1$?
"""
    ),
    answer(
        r"""
**1.** $\sigma^2\frac{p}{n-p-1}=\sigma^2\frac{p/n}{1-p/n-1/n}\to\sigma^2\frac{\gamma}{1-\gamma}$. The empirical spectrum of $X^\top X/n$ converges to the Marchenko–Pastur law on $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (quoted). The risk is $\frac{\sigma^2}{n}\sum_i1/\lambda_i(X^\top X/n)$, and as $\gamma\to1$ the lower edge tends to $0$, so the smallest eigenvalues blow up the sum. Consistently, $\frac1p\mathrm{Tr}(X^\top X/n)^{-1}\to\frac1{1-\gamma}$.

**2.** Zero: OLS is unbiased whenever $X$ has full column rank (1.B.1). Below the threshold the whole risk is variance.
"""
    ),
    markdown(
        r"""
### 3.B. Above the threshold: minimum-norm interpolation

With more parameters than samples, least squares no longer has a unique
solution: infinitely many coefficient vectors fit the training data exactly.
Which one does training pick, and what is its risk? This is the right half of
the double-descent curve.

#### 3.B.1. The right branch, bias and variance

For $p>n$, $X$ has full row rank almost surely, and infinitely many $b$ interpolate the data ($Xb=y$). $X^+$ denotes the Moore–Penrose pseudoinverse.

1. Show that the minimum-norm least-squares solution is $\hat\beta=X^+y=X^\top(XX^\top)^{-1}y$ and that it interpolates. It is also the $\lambda\to0^+$ limit of ridge, and the limit of gradient descent from $0$ (1.D.4.4).
2. Let $P=X^+X$ be the orthogonal projection onto the row space of $X$. Show that $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$, with the two terms orthogonal. So $\mathcal R=\underbrace{\mathbb E\|(I-P)\beta\|^2}_{\text{conditional bias}^2}+\underbrace{\sigma^2\,\mathbb E\,\mathrm{Tr}(XX^\top)^{-1}}_{\text{variance}}$.
3. *Bias.* Show that $\mathbb E P=(n/p)I$, so that $\text{bias}^2=r^2(1-n/p)\to r^2(1-1/\gamma)$.
4. *Variance.* $XX^\top$ is an $n\times n$ Wishart matrix with $p$ degrees of freedom. Reuse 1.B.2 with the roles of $n$ and $p$ swapped to get $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$.
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

**4.** For $p>n+1$, by 1.B.2.1 with $(n,p)\to(p,n)$: $\mathbb E(XX^\top)^{-1}=I_n/(p-n-1)$, so the variance is $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$. (Hastie et al. state this branch as part of their Theorem 1. In the isotropic Gaussian case it is a two-line consequence of the same Wishart fact.)

**5.**

$$
\mathcal R(\gamma)=\begin{cases}\sigma^2\dfrac{\gamma}{1-\gamma},&\gamma<1,\\[2mm] r^2\Big(1-\dfrac1\gamma\Big)+\dfrac{\sigma^2}{\gamma-1},&\gamma>1.\end{cases}
$$

The risk starts at $0$, rises, and diverges at $\gamma=1$ from both sides; the divergence is entirely variance. Past the threshold, the bias grows from $0$ toward $r^2$ while the variance falls from $\infty$ toward $0$. As $\gamma\to\infty$ the risk tends to $r^2$, the null risk: the projection captures a vanishing fraction of $\beta$, and the noise is spread over so many directions that it does no harm. The "second descent" is the variance falling faster than the bias rises.
"""
    ),
    markdown(
        r"""
### 3.C. The second descent and the cure

Past the peak the min-norm risk falls again. Does it ever fall below the risk
of predicting zero, and is the peak unavoidable, or an artefact of not
regularizing?

#### 3.C.1. When does interpolation beat doing nothing, and what removes the peak?

Two facts from §1 are used. Ridge regression,
$\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$, is the posterior mean
of $\beta$ under the Gaussian prior $\beta\sim\mathcal N(0,\tau^2I)$ and
Gaussian noise when $\lambda=\sigma^2/(n\tau^2)$ (1.D.1.1); and the posterior
mean minimizes the expected squared error averaged over the prior. The
minimum-norm interpolant is the $\lambda\to0^+$ limit of ridge. Write
$\lambda^*=\sigma^2\gamma/r^2$; part 2 shows it is the optimal ridge penalty in
this model ("optimally tuned ridge").

*Source for 3:* [Nakkiran, Venkat, Kakade and Ma (2020)](https://arxiv.org/abs/2003.01897)
prove that optimally tuned ridge is monotone in sample size and model size in
isotropic linear regression.

1. For $\gamma>1$, minimize $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$. Show that if $r>\sigma$ (signal-to-noise ratio $r^2/\sigma^2>1$), the minimum is at $\gamma=r/(r-\sigma)$, with value $2r\sigma-\sigma^2<r^2$. Show that if $r\le\sigma$, the risk decreases monotonically toward $r^2$ from above.
2. Show that $\lambda^*=\sigma^2\gamma/r^2$ is 1.D.1's $\lambda=\sigma^2/(n\tau^2)$ for the prior $\beta\sim\mathcal N(0,(r^2/p)I)$, which has $\mathbb E\|\beta\|^2=r^2$. Hence ridge at $\lambda^*$ is the posterior mean, the Bayes-optimal estimator under squared loss for that prior.
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

**2.** With $\tau^2=r^2/p$: $\lambda=\frac{\sigma^2}{n\tau^2}=\frac{\sigma^2p}{nr^2}=\frac{\sigma^2\gamma}{r^2}=\lambda^*$. By 1.D.1.1, ridge at $\lambda^*$ is the posterior mean, which minimizes $\mathbb E\|b-\beta\|^2$ over all estimators under that prior. (This is also 1.C.2.3's best common penalty $n\lambda=\sigma^2/\tau^2$.)

**3.** (a) Ridge and min-norm are both rotation-equivariant, and the design is rotation-invariant, so each one's risk at fixed $\beta$ depends only on $\|\beta\|$. Averaged over the prior, ridge($\lambda^*$) is no worse than any estimator. For either of these linear estimators, the noise- and design-averaged risk has the form $A\|\beta\|^2+B$: isotropy makes the squared-bias matrix a scalar multiple of $I$, and the noise variance is independent of $\beta$. Thus averaging over a prior with $\mathbb E\|\beta\|^2=r^2$ gives exactly its risk at any fixed vector of norm $r$, already at finite $p$ (when the risks are finite). The Bayes comparison therefore gives the fixed-radius comparison too. This does not make ridge Bayes-optimal under the uniform-on-a-sphere prior among all estimators.

(b) A Bayes estimator given $n+1$ samples can ignore the last one, so its Bayes risk is at most that of the Bayes estimator with $n$ samples. More data never hurts the posterior mean. The limiting curve depends on $\gamma$ alone, and at fixed $p$ increasing $n$ decreases $\gamma$, so the optimal-ridge risk is non-decreasing in $\gamma$. There is no peak.

The double-descent peak is therefore a property of the minimum-norm estimator, a variance divergence at $p\approx n$ that properly tuned regularization removes, and not a property of the problem.
"""
    ),
    markdown(
        r"""
#### 3.C.2. The curves at two signal-to-noise ratios

For $r^2/\sigma^2=5$ and for $r^2/\sigma^2=0.5$, use 3.C.1.1 to locate the
min-norm risk's minimum on $\gamma>1$, if it has one, and say whether the curve
crosses the null risk $r^2$. Then sketch both curves against $\gamma$ on a log
axis, with $r^2$ as a horizontal line, the bias and variance contributions
shaded, and optimal ridge on the same axes. The interactive figure below draws
all of these.
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
        r"""
"""
        + figure(
            "assets/interactive/double_descent.html",
            "Turn on the bias/variance decomposition and check which term makes the peak. Toggle optimal "
            "ridge λ* = σ²γ/r². Drag the SNR across 1 and watch whether the min-norm curve dips below "
            "the null risk r². The dots are finite-n simulations: note how they scatter near γ = 1.",
        )
    ),
]

S3_EF = [
    markdown(
        r"""
### 3.E. What regime are we in?

**The question.** Is the proportional regime of this section, $p/n=O(1)$, the
one modern models are in? Classical asymptotics (and all of §4) fix $p$ and
send $n\to\infty$; 3.A–3.D let $p$ and $n$ grow together.
[Chinchilla](https://arxiv.org/abs/2203.15556)'s compute-optimal allocation, of
roughly 20 training tokens per parameter, puts LLM pretraining at
$p/n\approx1/20$ if a token counted as a sample: neither $p\ll n$ nor
$p\gg n$. Two cautions. Tokens are not i.i.d. regression samples. And
pretraining usually runs for about one epoch, so the training loss is never
driven to zero, and nothing interpolates. Networks show double-descent-like
curves in width, training time and sample size
([Nakkiran et al. 2019](https://arxiv.org/abs/1912.02292)), typically with
label noise, and these are weakened by regularization and early stopping. That
phenomenology is consistent with the variance mechanism derived here, but it is
not derived from it.

#### 3.E.1. Question

Using 1.C.1, describe a measurement that would tell you whether a network's
double-descent peak is a variance divergence of the kind derived here.
"""
    ),
    answer(
        r"""
Train several networks at the same size on independent data draws (or with different seeds and different label noise) near the apparent peak. Decompose the test loss into the loss of the ensemble average (bias plus noise) and the spread across the ensemble (variance), as in 1.C.1. If the peak is a variance divergence, the spread peaks there while the ensemble-average loss does not. Weight decay tuned per size should then flatten the peak. A second check is the spectrum of the relevant kernel or feature covariance at the peak: there should be near-zero eigenvalues.
"""
    ),
    markdown(
        r"""
### 3.F. Readings

"""
        + reading(
            "Surprises in high-dimensional ridgeless least squares interpolation",
            "https://arxiv.org/abs/1903.08560",
            "Trevor Hastie, Andrea Montanari, Saharon Rosset, Ryan J. Tibshirani (Annals of Statistics, 2022)",
            "Exact asymptotics, as p/n → γ, for the risk of minimum-norm interpolation and ridge, under "
            "isotropic and general covariance, in well-specified and misspecified linear models and a "
            "random-features model. Double descent appears in the simplest of these, the peak is a "
            "variance divergence, and optimally tuned ridge has no peak.",
            "Read §1 (overview and figures) and §3 (isotropic features: Theorem 1 is the formula you "
            "derived in 3.B, for general i.i.d. entries rather than only Gaussian ones). §5 on misspecified "
            "models shows when the global minimum can sit at γ > 1. §4 (general covariance) and §8 "
            "(nonlinear random features) are for those heading to WB5.",
            importance=3, length=3,
        )
    ),
]

