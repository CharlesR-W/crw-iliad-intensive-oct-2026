"""MDL1 §2 (reference cases: risk, least squares, bias–variance) and §3 (no free lunch,
universal approximation, VC dimension).

Formulas are checked by checks/wb2a_check.py and checks/wb2_new_pacbayes_check.py (VC part).
"""
from nbkit import *  # noqa: F401,F403

ESL = "https://hastie.su.domains/ElemStatLearn/"
UML = "http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning"

REFERENCE = [
    markdown(
        r"""
<a id="wb2-section-2"></a>

## 2. Reference cases: risk, least squares, bias–variance

The textbook quantities (generalization gap, excess risk, bias and variance),
computed exactly in linear regression, where claims about generalization can
be checked against a closed form.

Notation: $X$ is the $n\times p$ design with singular values $s_k$, $\varepsilon$ the label noise with variance $\sigma^2$, $\lambda$ the ridge penalty in $\frac1n\|y-X\beta\|^2+\lambda\|\beta\|^2$ and $a=n\lambda$, $\tau^2$ a Gaussian prior variance on $\beta$; $\mu=X\beta$ in 2.3.4 is the noiseless mean vector.
"""
        + references(
            book("The Elements of Statistical Learning", ESL,
                 "Hastie, Tibshirani and Friedman (2nd ed. 2009; free PDF)",
                 "§3.2, §3.4.1, §7.2–7.4"),
        )
        + r"""
### Exercises: generalization gap and excess risk

"Deep networks generalize well" is said about two different quantities, and
papers switch between them.

**Setup.** Data $z=(x,y)$ are drawn i.i.d. from an unknown $P$; the training
sample is $S=(z_1,\dots,z_n)$. A loss $\ell(\theta;z)$ scores parameters
$\theta$ on one example. The *population risk* is
$R(\theta)=\mathbb E_{z\sim P}\ell(\theta;z)$, the *empirical risk*
$\hat R_n(\theta)=\frac1n\sum_i\ell(\theta;z_i)$. A learning rule maps $S$ to
$\hat\theta_n$; *empirical risk minimization* (ERM) takes
$\hat\theta_n\in\arg\min\hat R_n$. The population minimizer is
$\theta^*\in\arg\min R$. For any learning rule define

$$
\mathrm{gap}_n=R(\hat\theta_n)-\hat R_n(\hat\theta_n),
\qquad
\mathcal E_n=R(\hat\theta_n)-R(\theta^*).
$$

The *generalization gap* asks whether the training loss is an honest estimate
of the test loss; the *excess risk* asks whether the learner found a good model.

**2.1. Gap without excess, excess without gap.**

1. Construct a model class and a distribution in which ERM has excess risk exactly zero but a gap equal to the noise variance.
2. Construct a learning rule with expected gap exactly zero but excess risk bounded away from zero.
3. Show that for ERM, $\mathbb E\,\mathcal E_n\le\mathbb E\,\mathrm{gap}_n$, and that $\mathcal E_n\le 2\sup_\theta|R(\theta)-\hat R_n(\theta)|$.
4. You have a held-out set of $m$ fresh samples from $P$. Which of the two quantities can you estimate, and how accurately? What would you need in order to estimate the other?

<details><summary>Hint</summary>

For 1, use squared loss with a continuous input, and let the class contain functions that agree with the best predictor except on a finite set of points. For 2, the rule need not look at the data. For 3, telescope $\mathcal E_n$ through $\hat R_n(\hat\theta_n)$ and $\hat R_n(\theta^*)$, and ask which terms depend on $S$ in a way that matters.
</details>
"""
    ),
    answer(
        r"""
**1.** Take $x\sim\mathrm{Unif}[0,1]$ and $y=f^*(x)+\varepsilon$ with $\varepsilon\sim\mathcal N(0,\sigma^2)$, under squared loss. Let the class be $f^*$ together with every "patched" function $f_T$, where $T$ is a finite set of labelled points with distinct inputs: $f_T(x)=y$ if $(x,y)\in T$ and $f_T(x)=f^*(x)$ otherwise. Each $f_T$ differs from $f^*$ on a set of measure zero, so $R(f_T)=R(f^*)=\sigma^2$ and every member of the class is a population minimizer. ERM returns $f_S$, which has zero training loss because the $x_i$ are almost surely distinct. Hence

$$
\mathcal E_n=0,\qquad \mathrm{gap}_n=\sigma^2 .
$$

Memorizing costs nothing here because the memorized inputs never recur. This is a caricature of benign overfitting; 5.2–5.3 have a real version.

**2.** Take the rule that ignores the data and outputs a fixed $\theta_0$ with $R(\theta_0)>R(\theta^*)$. Since $\theta_0$ does not depend on $S$, $\mathbb E\hat R_n(\theta_0)=R(\theta_0)$, so $\mathbb E\,\mathrm{gap}_n=0$. Meanwhile $\mathcal E_n=R(\theta_0)-R(\theta^*)>0$ for every $S$. A small gap means the training loss is an honest estimate of the test loss, not that the model is good.

**3.** Write

$$
\mathcal E_n=\underbrace{R(\hat\theta_n)-\hat R_n(\hat\theta_n)}_{\mathrm{gap}_n}
+\underbrace{\hat R_n(\hat\theta_n)-\hat R_n(\theta^*)}_{\le 0\ \text{(ERM)}}
+\underbrace{\hat R_n(\theta^*)-R(\theta^*)}_{\text{mean }0}.
$$

The middle term is non-positive because $\hat\theta_n$ minimizes $\hat R_n$. The last has mean zero because $\theta^*$ does not depend on $S$. Taking expectations gives $\mathbb E\,\mathcal E_n\le\mathbb E\,\mathrm{gap}_n$. Bounding the first and third terms by $\sup_\theta|R-\hat R_n|$ and dropping the middle one gives $\mathcal E_n\le2\sup_\theta|R(\theta)-\hat R_n(\theta)|$. This is the route every capacity bound takes (§3, 1.1): control the worst gap over the whole class, and the excess risk follows.

**4.** Because $\hat\theta_n$ is independent of the held-out data, the held-out mean loss is unbiased for $R(\hat\theta_n)$, with standard error $\mathrm{sd}(\ell)/\sqrt m$. Subtracting the training loss estimates the gap. The excess risk also needs $R(\theta^*)$, the best achievable loss in the class, which a held-out loss alone does not identify without further assumptions or knowledge. You can measure *differences* of excess risk between two fitted predictors in the same model class, since $R(\theta^*)$ cancels. Different classes can have different optimum risks. More generally, a held-out set estimates the population loss of whatever predictor you evaluate on it, whether one network or an ensemble.

A sanity example with both quantities nonzero: for constant predictors on $y\sim\mathcal N(\mu,\sigma^2)$, $\hat\theta_n=\bar y$, $\mathbb E\,\mathcal E_n=\sigma^2/n$ and $\mathbb E\,\mathrm{gap}_n=2\sigma^2/n$.
"""
    ),
    markdown(
        r"""
### Exercises: least squares, exactly

The "parameter counting" answer, about $\sigma^2/n$ of excess risk per
parameter, and where it breaks down as $p$ approaches $n$.

**Setup for 2.2–2.3.** The linear model is $y=x^\top\beta+\varepsilon$, with
$x,\beta\in\mathbb R^p$ and noise $\varepsilon$ of mean $0$ and variance
$\sigma^2$, independent of $x$. Stack the $n$ training inputs as the rows of
the design matrix $X\in\mathbb R^{n\times p}$ and the targets as
$y\in\mathbb R^n$, so $y=X\beta+\varepsilon$. Take
$\varepsilon\sim\mathcal N(0,\sigma^2I_n)$, $n>p$, and $X$ of full column rank.
Ordinary least squares (OLS) is ERM with $\hat R_n(b)=\frac1n\|y-Xb\|^2$.
For the population, take a fresh $x\sim\mathcal N(0,I_p)$. The thin SVD
$X=\sum_ks_ku_kv_k^\top$ has singular values $s_k>0$; directions $v_k$ with
small $s_k$ are the ones the data barely determine.

**2.2. Mean, covariance and excess risk at a fixed design.**

1. Derive $\hat\beta=(X^\top X)^{-1}X^\top y$, and show $\mathbb E\hat\beta=\beta$ and $\mathrm{Cov}\,\hat\beta=\sigma^2(X^\top X)^{-1}$. Here expectations are over the noise, with $X$ fixed.
2. Show that $R(b)=\sigma^2+\|b-\beta\|^2$. Hence $\theta^*=\beta$, $\mathcal E_n=\|\hat\beta-\beta\|^2$, and $\mathbb E[\mathcal E_n\mid X]=\sigma^2\,\mathrm{Tr}\,(X^\top X)^{-1}$.
3. Show that the expected training risk is $\mathbb E[\hat R_n(\hat\beta)\mid X]=\sigma^2(n-p)/n$.

<details><summary>Hint</summary>

For 3, the residual is $(I-P_X)\varepsilon$, where $P_X=X(X^\top X)^{-1}X^\top$ is a rank-$p$ orthogonal projection.
</details>
"""
    ),
    answer(
        r"""
**1.** $\nabla_b\hat R_n=-\frac2nX^\top(y-Xb)=0$ gives the normal equations $X^\top Xb=X^\top y$. Since $X^\top X$ is invertible, $\hat\beta=(X^\top X)^{-1}X^\top y$. Substituting $y=X\beta+\varepsilon$,

$$
\hat\beta=\beta+(X^\top X)^{-1}X^\top\varepsilon,
$$

so $\mathbb E\hat\beta=\beta$, and

$$
\mathrm{Cov}\,\hat\beta=(X^\top X)^{-1}X^\top(\sigma^2I)X(X^\top X)^{-1}=\sigma^2(X^\top X)^{-1}.
$$

**2.** $R(b)=\mathbb E(x^\top\beta+\varepsilon-x^\top b)^2=\sigma^2+(\beta-b)^\top\mathbb E[xx^\top]\,(\beta-b)=\sigma^2+\|b-\beta\|^2$, since $\varepsilon$ is independent of $x$ and $\mathbb E xx^\top=I$. So $\theta^*=\beta$, $R(\theta^*)=\sigma^2$ and $\mathcal E_n=\|\hat\beta-\beta\|^2$. Then

$$
\mathbb E\|\hat\beta-\beta\|^2=\mathrm{Tr}\,\mathrm{Cov}\,\hat\beta=\sigma^2\,\mathrm{Tr}\,(X^\top X)^{-1}.
$$

**3.** The residual is $y-X\hat\beta=(I-P_X)y=(I-P_X)\varepsilon$, because $(I-P_X)X=0$. So $\mathbb E\|y-X\hat\beta\|^2=\sigma^2\,\mathrm{Tr}(I-P_X)=\sigma^2(n-p)$, and dividing by $n$ gives $\sigma^2(n-p)/n$. The fit absorbs the noise along $p$ directions, so the training loss *underestimates* $\sigma^2$.
"""
    ),
    markdown(
        r"""
**2.3. Random design: parameter counting.** Now let the rows of $X$ be i.i.d.
$\mathcal N(0,I_p)$, so $X^\top X$ is Wishart.

1. Show that $\mathbb E[(X^\top X)^{-1}]=I_p/(n-p-1)$ for $n>p+1$. Conclude that $\mathbb E\,\mathcal E_n=\sigma^2p/(n-p-1)$.
2. Compute the expected gap exactly, and show that for $n\gg p$ the excess risk is $\approx\sigma^2p/n$ while the gap is $\approx2\sigma^2p/n$. Where does the factor of two come from?
3. What happens as $p\to n-1$? Which eigenvalues of $X^\top X$ are responsible?

<details><summary>Hint</summary>

Rotation invariance shows that $\mathbb E(X^\top X)^{-1}$ is a multiple of $I$. For a diagonal entry, the block-inverse (Schur complement) formula gives $[(X^\top X)^{-1}]_{11}=1/\big(x_{\cdot1}^\top(I-P_{-1})x_{\cdot1}\big)$. Here $x_{\cdot1}\in\mathbb R^n$ is the first column of $X$ and $P_{-1}$ projects onto the span of the other $p-1$ columns. What is the law of the denominator? You may use $\mathbb E[1/\chi^2_k]=1/(k-2)$.
</details>
"""
    ),
    answer(
        r"""
**1.** *Proportional to $I$.* For any orthogonal $O\in\mathbb R^{p\times p}$, $XO$ has the same law as $X$, and $((XO)^\top XO)^{-1}=O^\top(X^\top X)^{-1}O$. So $M=\mathbb E(X^\top X)^{-1}$ satisfies $M=O^\top MO$ for all $O$, which forces $M=cI$.

*The constant.* By the Schur complement, $[(X^\top X)^{-1}]_{11}=1/\big(x_{\cdot1}^\top(I-P_{-1})x_{\cdot1}\big)$. Conditional on the other columns, $I-P_{-1}$ is an orthogonal projection of rank $n-p+1$, and it is independent of $x_{\cdot1}\sim\mathcal N(0,I_n)$. So the denominator is $\chi^2_{n-p+1}$. Next, $\mathbb E[1/\chi^2_k]=\frac{\Gamma(k/2-1)}{2\Gamma(k/2)}=\frac1{k-2}$ for $k>2$, which gives $c=1/(n-p-1)$. Combining with 2.2.2,

$$
\mathbb E\,\mathcal E_n=\sigma^2\,\mathrm{Tr}\,\mathbb E(X^\top X)^{-1}=\frac{\sigma^2p}{n-p-1}.
$$

**2.** $\mathbb E R(\hat\beta)=\sigma^2+\sigma^2p/(n-p-1)=\sigma^2\frac{n-1}{n-p-1}$, and by 2.2.3, $\mathbb E\hat R_n(\hat\beta)=\sigma^2\frac{n-p}{n}$. So

$$
\mathbb E\,\mathrm{gap}_n=\sigma^2\Big[\frac{n-1}{n-p-1}-\frac{n-p}{n}\Big]
\approx\sigma^2\Big[\Big(1+\frac pn\Big)-\Big(1-\frac pn\Big)\Big]=\frac{2\sigma^2p}{n}.
$$

The test loss *overestimates* $\sigma^2$ by about $\sigma^2p/n$ (the excess risk), and the training loss *underestimates* it by the same amount, because the fit has absorbed the noise along $p$ directions. The gap counts both. This is the factor of two between "optimism" and "excess risk" that 7.2 derives in general (TIC vs $\tfrac1{2n}\mathrm{Tr}(H^{-1}C)$).

**3.** Each of the $p$ coefficients is estimated with variance $\approx\sigma^2/n$, which is "parameter counting". The exact correction $n/(n-p-1)$ diverges as $p\to n-1$. The cause is the smallest eigenvalues of $X^\top X/n$. When $p/n\to\gamma<1$, its spectrum fills $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (Marchenko–Pastur, quoted), and $\mathrm{Tr}(X^\top X)^{-1}=\frac1n\sum_i 1/\lambda_i(X^\top X/n)$ is dominated by the eigenvalues near the lower edge, which tends to $0$ as $\gamma\to1$. Noise along nearly degenerate directions of the design is amplified without limit. The keystone exercises (1.3) pick this up.

(All of these formulas are checked by Monte Carlo in `checks/wb2a_check.py`.)
"""
    ),
    markdown(
        r"""
### Exercises: bias–variance

Why a model that fits its training data worse can predict better, and where
the decomposition stops being true.

**2.4. Decompose, then decompose a posterior.** Let $y=f(x)+\varepsilon$,
where $f$ is the true regression function, $\mathbb E[\varepsilon\mid x]=0$ and
$\mathrm{Var}[\varepsilon\mid x]=\sigma^2(x)$. A predictor $\hat f_S$ is
trained on a random sample $S$, independent of the test point. Fix a test
input $x$; expectations are over $S$ and the test noise.

1. Show that $\mathbb E_{S,\varepsilon}\big(\hat f_S(x)-y\big)^2=\sigma^2(x)+\big(\mathbb E_S\hat f_S(x)-f(x)\big)^2+\mathrm{Var}_S\hat f_S(x)$.
2. Average $K$ predictors trained on independent samples. Which terms change, and by how much?
3. The same algebra with a different source of randomness. Let $q$ be any distribution over parameters, for instance a posterior given $S$, with predictive mean $\bar f(x)=\mathbb E_{\theta\sim q}f_\theta(x)$. Show that for squared loss
$$\mathbb E_{\theta\sim q}\big(f_\theta(x)-y\big)^2=\big(\bar f(x)-y\big)^2+\mathrm{Var}_{\theta\sim q}f_\theta(x).$$
Translate this into a statement comparing a single posterior draw with the posterior-predictive mean. (7.6 calls their errors the Gibbs and Bayes errors.)
4. Does 1 hold for 0–1 loss? Give an example where *more* variance lowers the expected 0–1 error.
"""
    ),
    answer(
        r"""
**1.** Write $\hat f_S(x)-y=(\hat f_S-\mathbb E\hat f_S)+(\mathbb E\hat f_S-f)+(f-y)$ and square. The cross terms vanish in expectation. $(\hat f_S-\mathbb E\hat f_S)$ has mean zero over $S$. $(f-y)=-\varepsilon$ has mean zero and is independent of $S$. $(\mathbb E\hat f_S-f)$ is a constant. What remains is variance, plus squared bias, plus $\sigma^2(x)$.

**2.** The average of $K$ independent copies has the same mean, hence the same bias, and variance $\mathrm{Var}_S\hat f_S(x)/K$. The noise term is untouched. (Real bagging resamples a single dataset, so the copies are correlated and the reduction is smaller.)

**3.** Expand $f_\theta-y=(f_\theta-\bar f)+(\bar f-y)$ and take $\mathbb E_{\theta\sim q}$ with $x,y$ fixed. The cross term is $2(\bar f-y)\,\mathbb E_q[f_\theta-\bar f]=0$. With $q$ a posterior, the left side is the Gibbs loss at $(x,y)$ and $(\bar f-y)^2$ is the Bayes-ensemble loss. Averaging over $(x,y)$:

$$
\text{Gibbs error}=\text{Bayes-ensemble error}+\mathbb E_x\,\mathrm{Var}_{\text{post}}f_\theta(x)\ \ge\ \text{Bayes-ensemble error},
$$

with equality iff the posterior predictors agree for almost every input under the test distribution. For squared loss, ensembling never hurts a posterior draw. 7.6 finds that in the regular case the excess parts differ by a factor of two ($p/n$ vs $p/2n$).

**4.** No: the 0–1 loss does not decompose additively. Take $y=1$ deterministically and predict $\mathbf 1[\hat g_S(x)>0]$, where $\hat g_S(x)\sim\mathcal N(\mu,v)$ with $\mu<0$, so the average prediction is on the wrong side. The error is $\Pr[\hat g_S\le0]=\Phi(-\mu/\sqrt v)$, which *decreases* toward $\tfrac12$ as $v$ grows. When the bias has the wrong sign, variance helps. "Variance always hurts" is a fact about squared loss.
"""
    ),
    markdown(
        r"""
**2.5. Ridge in the eigenbasis: bias and variance per direction.** Take the
linear model $y=X\beta+\varepsilon$ with a fixed $n\times p$ design $X$ and
noise $\varepsilon\sim(0,\sigma^2I_n)$, $\sigma^2>0$. For the comparisons with
OLS assume $X$ has full column rank. *Ridge regression* is

$$
\hat\beta_\lambda=\arg\min_b\ \tfrac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y .
$$

Write $X^\top X=V\,\mathrm{diag}(s_1^2,\dots,s_p^2)\,V^\top$, with $v_k$ the
columns of $V$, $\beta_k=v_k^\top\beta$, and $a=n\lambda$. Track the parameter
error $\mathbb E\|\hat\beta_\lambda-\beta\|^2$, which equals the excess
population risk when test inputs are $x\sim\mathcal N(0,I)$ (2.2.2).

1. Show that, in coordinates $v_k^\top\hat\beta_\lambda$,
$$\mathrm{bias}_k=-\frac{a}{s_k^2+a}\,\beta_k,\qquad \mathrm{var}_k=\sigma^2\frac{s_k^2}{(s_k^2+a)^2},$$
and that different coordinates are uncorrelated. Hence $\mathbb E\|\hat\beta_\lambda-\beta\|^2=\sum_k\dfrac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$.
2. Which directions carry most of the variance at $\lambda=0$? Show that the derivative of the total error with respect to $a$ at $a=0$ is strictly negative, so some ridge always beats OLS.
3. Allow a separate penalty $a_k$ in each direction. Show that the optimum is $a_k=\sigma^2/\beta_k^2$, independent of $s_k$, and that the minimal error in that direction is $\big(s_k^2/\sigma^2+1/\beta_k^2\big)^{-1}$. If $\beta\sim\mathcal N(0,\tau^2I)$, show that the best *common* penalty is $a=\sigma^2/\tau^2$.
4. *Effective degrees of freedom.* The fitted values are $\hat y=S_\lambda y$ with $S_\lambda=X(X^\top X+aI)^{-1}X^\top$. Show that $\mathrm{Tr}\,S_\lambda=\mathrm{df}(\lambda)=\sum_k s_k^2/(s_k^2+a)$. Then show that the in-sample optimism, $\mathbb E\frac1n\|y'-\hat y\|^2-\mathbb E\frac1n\|y-\hat y\|^2$ for fresh noise $y'=X\beta+\varepsilon'$ at the same inputs, equals $2\sigma^2\,\mathrm{df}(\lambda)/n$. Compare with 2.3.

<details><summary>Hint</summary>

For 1, $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, and the covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}$. All of these matrices are diagonal in $V$. For 3, differentiate $f(a)=(a^2\beta^2+\sigma^2s^2)/(s^2+a)^2$; the numerator of $f'$ factors. For 4, expand both squared norms around $X\beta$. The only surviving difference is $\mathbb E\langle\varepsilon,S_\lambda\varepsilon\rangle$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, so in the $V$ basis coordinate $k$ is $\frac{s_k^2}{s_k^2+a}\beta_k$ and the bias is $\big(\frac{s_k^2}{s_k^2+a}-1\big)\beta_k=-\frac{a}{s_k^2+a}\beta_k$. The covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}=V\,\mathrm{diag}\!\big(\sigma^2s_k^2/(s_k^2+a)^2\big)V^\top$, which is diagonal in $V$, so the coordinates are uncorrelated. Summing squared bias and variance over $k$ gives the total.

**2.** At $a=0$, $\mathrm{var}_k=\sigma^2/s_k^2$, so the weakest directions (small $s_k$) dominate. That is the same $\mathrm{Tr}(X^\top X)^{-1}$ as in 2.2.2. Write $f_k(a)=\frac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$. Then

$$
f_k'(a)=\frac{2a\beta_k^2(s_k^2+a)-2(a^2\beta_k^2+\sigma^2s_k^2)}{(s_k^2+a)^3}=\frac{2s_k^2(a\beta_k^2-\sigma^2)}{(s_k^2+a)^3}.
$$

At $a=0$ this is $-2\sigma^2/s_k^4<0$ for every $k$, so a small amount of ridge lowers the total error, for every $\beta$. This is the observation of [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634).

**3.** By the factored derivative, $f_k$ decreases for $a<\sigma^2/\beta_k^2$ and increases after, so $a_k^*=\sigma^2/\beta_k^2$, which does not involve $s_k$. Substituting,

$$
f_k(a_k^*)=\frac{\sigma^2(\sigma^2+s_k^2\beta_k^2)/\beta_k^2}{(s_k^2\beta_k^2+\sigma^2)^2/\beta_k^4}=\frac{\sigma^2\beta_k^2}{s_k^2\beta_k^2+\sigma^2}=\Big(\frac{s_k^2}{\sigma^2}+\frac1{\beta_k^2}\Big)^{-1}.
$$

If $\beta_k=0$, the optimum is the limit $a_k\to\infty$, with zero error. This has the form of a posterior variance: data precision plus prior precision. If $\beta\sim\mathcal N(0,\tau^2I)$, then $\mathbb E\beta_k^2=\tau^2$, and $\mathbb E_\beta f_k(a)=\frac{a^2\tau^2+\sigma^2s_k^2}{(s_k^2+a)^2}$ is minimized at $a=\sigma^2/\tau^2$ for every $k$ at once. So the best common penalty is $n\lambda=\sigma^2/\tau^2$. 4.1 gets the same value as the MAP prior, and 1.5.2 as the optimal $\lambda^*=\sigma^2\gamma/r^2$ for random designs, where $\gamma=p/n$, $r^2=\|\beta\|^2$ and $\tau^2=r^2/p$.

**4.** With the thin SVD $X=U\,\mathrm{diag}(s_k)V^\top$, $S_\lambda=U\,\mathrm{diag}\big(s_k^2/(s_k^2+a)\big)U^\top$, so $\mathrm{Tr}\,S_\lambda=\sum_k s_k^2/(s_k^2+a)\in[0,p]$. Let $\mu=X\beta$. Then

$$
\|y'-\hat y\|^2=\|\varepsilon'\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon',\mu-\hat y\rangle,\qquad
\|y-\hat y\|^2=\|\varepsilon\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon,\mu-\hat y\rangle .
$$

The first cross term has mean $0$ because $\varepsilon'$ is independent of $\hat y$. In the second, $\mu-\hat y=(I-S_\lambda)\mu-S_\lambda\varepsilon$, so $\mathbb E\langle\varepsilon,\mu-\hat y\rangle=-\sigma^2\,\mathrm{Tr}\,S_\lambda$. The difference is $2\sigma^2\,\mathrm{Tr}\,S_\lambda$, and dividing by $n$ gives $2\sigma^2\mathrm{df}(\lambda)/n$. At $\lambda=0$, $\mathrm{df}=p$, which has the same leading $2\sigma^2p/n$ value as 2.3. Here it is exact for new labels at the training inputs; 2.3 evaluates fresh random inputs, so its finite-sample gap differs. Ridge replaces "count the parameters" with "count the directions the data can resolve against the penalty."
"""
    ),
]

THEOREMS = [
    markdown(
        r"""
<a id="wb2-section-3"></a>

## 3. Theorems that seem to forbid learning

No free lunch, universal approximation and capacity bounds each seem to say
deep learning should be impossible or unexplained; each is true, and the
exercises find the assumption that keeps it from biting.

Notation: $d$ is the input dimension, $d_{\mathrm{VC}}$ the VC dimension, $m$ a number of points; the $\alpha_i$ in 3.3 are Radon coefficients. No ridge penalty appears in this section.
"""
        + reading(
            "The lack of a priori distinctions between learning algorithms",
            "https://doi.org/10.1162/neco.1996.8.7.1341",
            "David H. Wolpert (Neural Computation, 1996)",
            "The original off-training-set no-free-lunch theorem: averaged uniformly over targets, all learners "
            "are equal. 3.1 proves the core of it.",
            importance=2, length=3,
        )
        + reading(
            "Representation benefits of deep feedforward networks",
            "https://arxiv.org/abs/1509.08101",
            "Matus Telgarsky (2015)",
            "A short depth-separation construction: composing a tent map gives exponentially many linear pieces "
            "that shallow networks need exponentially many units to match (3.2.4).",
            importance=3, length=1,
        )
        + reading(
            "Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks",
            "https://jmlr.org/papers/v20/17-612.html",
            "Peter L. Bartlett, Nick Harvey, Christopher Liaw, Abbas Mehrabian (JMLR, 2019)",
            "The VC dimension of ReLU networks with W weights in L layers is O(WL log W), and some architectures "
            "reach Ω(WL log(W/L)). Read the introduction for the statements.",
            importance=2, length=3,
        )
        + r"""
Also: [Leshno, Lin, Pinkus and Schocken (1993)](https://linkinghub.elsevier.com/retrieve/pii/S0893608005801315), universal approximation holds exactly for non-polynomial activations.
"""
        + references(
            book("Understanding Machine Learning: From Theory to Algorithms", UML,
                 "Shalev-Shwartz and Ben-David (2014; free PDF)",
                 "§2.3, ch. 4–6, §7.3"),
        )
        + r"""
### Exercises: no free lunch

**3.1. Prove NFL, then find the bet.** Let the input domain $\mathcal X$ be
finite, $|\mathcal X|=N$, and the target a binary function
$f:\mathcal X\to\{0,1\}$. A training set consists of inputs
$x_1,\dots,x_n$, chosen independently of $f$, with labels $f(x_i)$; at least
one input remains unseen. A learner is any map, possibly randomized, from the
labelled sample to a hypothesis $h:\mathcal X\to\{0,1\}$. With $U$ the set of
unseen inputs, the off-training-set error is
$\mathrm{err}_{\mathrm{OTS}}(h,f)=\frac1{|U|}\sum_{x\in U}\mathbf 1[h(x)\ne f(x)]$.

1. Let $f$ be uniform over all $2^N$ functions. Prove that every learner has expected off-training-set error exactly $\tfrac12$, conditional on any training inputs.
2. Conclude that for any two learners $A$ and $B$, $\sum_f\big[\mathrm{err}_{\mathrm{OTS}}(A,f)-\mathrm{err}_{\mathrm{OTS}}(B,f)\big]=0$. What does this say about a learner that beats another on some targets?
3. Now let $f$ lie in a class $\mathcal F$ with $|\mathcal F|=M$, and let the inputs be i.i.d. from a distribution $D$ on $\mathcal X$. Show that a learner returning any $h\in\mathcal F$ *consistent* with the sample (agreeing with $f$ on every training input) satisfies $\Pr[\mathrm{err}_D(h)>\epsilon]\le Me^{-n\epsilon}$, where $\mathrm{err}_D$ is the error under $D$. How many samples suffice if $\mathcal F$ is the set of functions computed by programs of at most $b$ bits? What does the bound give for the class of all functions?
4. Where in the proof of 1 did the uniform prior enter? Give priors over targets under which the training labels are informative about the unseen ones, and say which kind of learner each favours.

<details><summary>Hint</summary>

For 1, condition on the training labels. Under the uniform prior, the values $f(x)$ are independent fair coins, so what is the conditional law of $f$ on $U$? For 3, a fixed $h$ with error above $\epsilon$ survives $n$ i.i.d. samples with probability at most $(1-\epsilon)^n$. Then take a union bound.
</details>
"""
    ),
    answer(
        r"""
**1.** Under the uniform prior, the values $\{f(x)\}_{x\in\mathcal X}$ are i.i.d. fair coins. Condition on the training inputs and their labels. The values $\{f(x)\}_{x\in U}$ are still i.i.d. fair coins, independent of everything the learner sees. The hypothesis $h$ is a function of the training labels and of independent internal randomness, so $h$ is independent of $f|_U$. Hence $\Pr[h(x)\ne f(x)]=\tfrac12$ for each $x\in U$, and the average over $U$ has expectation $\tfrac12$.

**2.** By 1, $\sum_f\mathrm{err}_{\mathrm{OTS}}(A,f)=2^N\cdot\tfrac12=\sum_f\mathrm{err}_{\mathrm{OTS}}(B,f)$, so the difference sums to zero. Whatever advantage $A$ has over $B$ on one set of targets, it gives back exactly on the complement. No learner is better than another "in general", even the learner that predicts the opposite of the majority label.

**3.** Fix $h$ with $\mathrm{err}_D(h)>\epsilon$. Each i.i.d. sample is consistent with it with probability below $1-\epsilon$, so $\Pr[h\text{ consistent with all }n]\le(1-\epsilon)^n\le e^{-n\epsilon}$. There are at most $M$ such $h$, so by the union bound $\Pr[\exists\text{ consistent }h\text{ with }\mathrm{err}_D(h)>\epsilon]\le Me^{-n\epsilon}$. Equivalently, with probability at least $1-\delta$, every consistent $h$ has $\mathrm{err}_D(h)\le(\ln M+\ln(1/\delta))/n$. For programs of at most $b$ bits, $M<2^{b+1}$, so $n\approx(b\ln2+\ln(1/\delta))/\epsilon$ samples suffice. The cost of learning scales with the *description length* of the target (Occam). For the class of all functions, $M=2^N$, and the bound needs $n\gtrsim N\ln2/\epsilon$: you must see a constant fraction of the domain, which is NFL again.

**4.** It entered at "$f|_U$ is independent of the labels on the training set". Any prior under which seen labels are informative about unseen ones escapes NFL. Examples:
- a simplicity (Solomonoff) prior, which favours the shortest consistent program (MDL3);
- a smoothness prior, which favours kernel or nearest-neighbour methods;
- a symmetry prior, which favours equivariant architectures such as CNNs (MDL2).

The question "why does deep learning generalize?" becomes: which prior over targets do the architecture and the optimizer implement, and why does it match the world?
"""
    ),
    markdown(
        r"""
### Exercises: universal approximation

**3.2. How many units for $\sin\omega x$?** The universal approximation
theorem (quoted): for any continuous $f$ on a compact $K\subset\mathbb R^d$
and any $\epsilon>0$, a one-hidden-layer network with a continuous
non-polynomial activation (for example ReLU) achieves $\sup_K|f-g|<\epsilon$.
It says nothing about how many units are needed.

1. Show that a continuous piecewise-linear function on $[0,1]$ with $K$ pieces is exactly a one-hidden-layer ReLU network with $K$ units plus an output bias.
2. *Upper bound.* Show that the piecewise-linear interpolant of $\sin(\omega x)$ at $K+1$ equally spaced knots has sup error at most $\omega^2/(8K^2)$. Deduce that $\lceil\omega/\sqrt{8\epsilon}\,\rceil$ units suffice for accuracy $\epsilon$.
3. *Lower bound.* Show that a one-hidden-layer ReLU network with $m$ units on $\mathbb R$ is piecewise linear with at most $m+1$ pieces. Then show that for $\epsilon<1$ accuracy, $m$ must grow at least like $\omega/(2\pi)$.
4. *Depth.* Let $t(x)=2\,\mathrm{ReLU}(x)-4\,\mathrm{ReLU}(x-\tfrac12)$, the tent map on $[0,1]$. Show that the $k$-fold composition $t^{\circ k}$ has $2^k$ linear pieces. Conclude that a depth-$k$ network with $2k$ units computes a function that any one-hidden-layer network needs at least $2^k-1$ units to represent.
5. List three things the theorem does not tell you.

<details><summary>Hint</summary>

For 2, the interpolation error on an interval $[a,b]$ is $e(x)=\tfrac12f''(\xi)(x-a)(x-b)$. For 3, the extrema of $\sin\omega x$ alternate between $+1$ and $-1$. On a piece where $g$ is affine, can $g$ be positive, negative, positive at three consecutive extrema? For 4, $t$ maps each half of $[0,1]$ monotonically onto $[0,1]$.
</details>
"""
    ),
    answer(
        r"""
**1.** Let the breakpoints be $0=t_0<t_1<\dots<t_{K-1}<1$, with slopes $a_0,\dots,a_{K-1}$ on the pieces. Then on $[0,1]$

$$
g(x)=g(0)+a_0\,\mathrm{ReLU}(x-t_0)+\sum_{k=1}^{K-1}(a_k-a_{k-1})\,\mathrm{ReLU}(x-t_k),
$$

since each ReLU switches on the slope change at its breakpoint. That is $K$ units plus a bias.

**2.** On a knot interval of length $h=1/K$, the interpolation error is $e(x)=\tfrac12f''(\xi)(x-a)(x-b)$, and $|(x-a)(x-b)|\le h^2/4$. With $|f''|\le\omega^2$ this gives $|e|\le\omega^2h^2/8=\omega^2/(8K^2)$. Setting $\omega^2/(8K^2)\le\epsilon$ gives $K\ge\omega/\sqrt{8\epsilon}$, so $m=\lceil\omega/\sqrt{8\epsilon}\rceil$ units suffice. The count is linear in the frequency.

**3.** Each unit $\mathrm{ReLU}(wx+b)$ has at most one breakpoint, so the sum has at most $m$ breakpoints and at most $m+1$ affine pieces. The extrema of $\sin\omega x$ in $[0,1]$ are at $x_j=(\pi/2+j\pi)/\omega$, with values $(-1)^j$, and there are $J=\lfloor\omega/\pi-\tfrac12\rfloor+1$ of them. If $\sup|g-\sin\omega x|<1$, then $g(x_j)$ has sign $(-1)^j$. An affine function that is positive at $x_j$ and $x_{j+2}$ is positive at $x_{j+1}$, which lies between them. So a single piece contains at most two consecutive extrema, and $J\le2(m+1)$, i.e. $m\ge J/2-1\approx\omega/(2\pi)$. Together with 2, the number of units needed is $\Theta(\omega)$ at fixed accuracy.

**4.** On $[0,\tfrac12]$, $t(x)=2x$ is an increasing bijection onto $[0,1]$; on $[\tfrac12,1]$, $t(x)=2-2x$ is a decreasing bijection onto $[0,1]$. So $t^{\circ k}=t^{\circ(k-1)}\circ t$ consists of two reparametrized copies of $t^{\circ(k-1)}$, one of them reversed, and by induction it has $2\cdot2^{k-1}=2^k$ pieces. Each layer uses 2 units, so the depth-$k$ network has $2k$ units. By 3, one hidden layer needs $m+1\ge2^k$ units. Depth buys pieces exponentially.

**5.** UAT does not tell you:
- *how many units* are needed. For generic Lipschitz targets in $d$ dimensions the count grows like $\epsilon^{-d}$.
- *whether gradient descent finds* the approximant.
- *how many samples* identify it, or *which* of the many interpolants of the data training selects.

Polynomials (Weierstrass) and lookup tables are universal approximators too. Expressivity is cheap, so it cannot be what distinguishes networks. UAT rules out "the net cannot represent the target" and nothing else.
"""
    ),
    markdown(
        r"""
### Exercises: VC dimension

The finite-class bound charges $\ln|\mathcal H|$, which is infinite for any
class with a real-valued parameter. What matters on $n$ points is how many
labellings the class produces, and that is not the number of parameters.

**3.3. Shattering and VC dimension.** Let $\mathcal H$ be a class of binary
classifiers $h:\mathcal X\to\{0,1\}$. A finite set $C=\{x_1,\dots,x_m\}$ is
*shattered* by $\mathcal H$ if all $2^m$ labellings of $C$ are produced by
some $h\in\mathcal H$. The *growth function* $\Pi_{\mathcal H}(m)$ is the
largest number of distinct labellings $\mathcal H$ produces on any $m$
points. The *VC dimension* $d_{\mathrm{VC}}(\mathcal H)$ is the largest $m$
for which some $m$-point set is shattered. To prove $d_{\mathrm{VC}}=d$,
exhibit one shattered set of $d$ points and show that no set of $d+1$ points
is shattered. Two facts are quoted. *Sauer–Shelah:* if
$d_{\mathrm{VC}}=d<\infty$, then $\Pi_{\mathcal H}(m)\le\sum_{i=0}^{d}\binom mi\le(em/d)^d$
for $m\ge d$. *VC bound* (Mohri, Rostamizadeh and Talwalkar, *Foundations of
Machine Learning*, 2nd ed., Cor. 3.19): with probability at least $1-\delta$,
for all $h\in\mathcal H$,
$R(h)\le\hat R_n(h)+\sqrt{2d\ln(en/d)/n}+\sqrt{\ln(1/\delta)/(2n)}$, which says
something only when $n\gg d$.

1. *Thresholds* on $\mathbb R$: $h_t(x)=\mathbf 1[x\ge t]$, $t\in\mathbb R$. Show $d_{\mathrm{VC}}=1$ and $\Pi(m)=m+1$.
2. *Intervals* on $\mathbb R$: $h_{a,b}(x)=\mathbf 1[a\le x\le b]$. Show $d_{\mathrm{VC}}=2$, compute $\Pi(m)$, and compare with the Sauer–Shelah bound.
3. *Affine halfspaces* in $\mathbb R^d$: $h_{w,b}(x)=\mathbf 1[w^\top x+b\ge0]$, with $d+1$ parameters. Show that the $d+1$ points $0,e_1,\dots,e_d$ are shattered. Then use *Radon's theorem* (any $d+2$ points in $\mathbb R^d$ can be split into two groups whose convex hulls intersect) to show that no $d+2$ points are shattered, so $d_{\mathrm{VC}}=d+1$. Optionally, prove Radon's theorem.
4. *One parameter, infinite VC dimension.* Let $h_\omega(x)=\mathbf 1[\sin(\omega x)>0]$ with $\omega>0$. Take $x_i=2^{-i}$ for $i=1,\dots,m$, any labels $y_1,\dots,y_m\in\{0,1\}$, and $\omega=\pi\big(1+\sum_{j=1}^m(1-y_j)2^j\big)$. Show that $h_\omega(x_i)=y_i$ for all $i$, so $d_{\mathrm{VC}}=\infty$. What does this say about counting parameters? What changes if $\omega$ is stored as a 32-bit float?

<details><summary>Hint</summary>

For 2, sort the points: which patterns of 0s and 1s can an interval cut out?
For 3, the halfspace $\{x:w^\top x+b\ge0\}$ and its complement are both
convex. For Radon, solve $\sum_i\alpha_ix_i=0$, $\sum_i\alpha_i=0$ and
split the points by the sign of $\alpha_i$. For 4, compute
$\omega x_i/\pi$ modulo 2: the terms with $j>i$ are even integers.
</details>
"""
    ),
    answer(
        r"""
**1.** One point $x$ is shattered: $t\le x$ labels it 1 and $t>x$ labels it 0. For two points $x_1<x_2$, the labelling $(1,0)$ needs $t\le x_1$ and $t>x_2$, which is impossible. So $d_{\mathrm{VC}}=1$. On $m$ distinct sorted points the achievable labellings are $0\cdots01\cdots1$, fixed by the number $k\in\{0,\dots,m\}$ of zeros. So $\Pi(m)=m+1=\binom m0+\binom m1$, and Sauer–Shelah holds with equality.

**2.** Two points $x_1<x_2$: a long interval gives $(1,1)$, a short interval around one point gives $(1,0)$ or $(0,1)$, and an interval away from both gives $(0,0)$. Three points $x_1<x_2<x_3$: an interval containing $x_1$ and $x_3$ contains $x_2$, so $(1,0,1)$ is impossible, and $d_{\mathrm{VC}}=2$. On $m$ sorted points the 1s must form one contiguous block. There is the empty block, plus one block for each choice of first and last point $i\le j$, which gives

$$
\Pi(m)=1+\frac{m(m+1)}2=\binom m0+\binom m1+\binom m2 ,
$$

again equal to the Sauer–Shelah bound.

**3.** *Lower bound.* Write the labels of $0,e_1,\dots,e_d$ as $y_0,\dots,y_d$ and let $s_i=2y_i-1\in\{\pm1\}$. Take $b=s_0/2$ and $w_i=s_i-s_0/2$. Then $w^\top0+b=s_0/2$ and $w^\top e_i+b=s_i$. Each has the sign $s_i$ and none is zero, so every labelling is produced and $d_{\mathrm{VC}}\ge d+1$.

*Upper bound.* Take any $d+2$ points $x_1,\dots,x_{d+2}$. The system $\sum_i\alpha_ix_i=0$, $\sum_i\alpha_i=0$ has $d+1$ equations and $d+2$ unknowns, so it has a solution $\alpha\ne0$. Because the $\alpha_i$ sum to zero, some are positive and some negative. Let $I=\{i:\alpha_i>0\}$, let $J$ be the rest, and let $s=\sum_{i\in I}\alpha_i>0$. Then

$$
p=\sum_{i\in I}\frac{\lambda_i}{s}x_i=\sum_{i\in J}\frac{-\lambda_i}{s}x_i
$$

is a convex combination of the $I$-points and also of the $J$-points (Radon's theorem). Label $I$ by 1 and $J$ by 0. A halfspace producing this labelling would have the convex set $\{w^\top x+b\ge0\}$ contain all $I$-points, hence $p$. The convex set $\{w^\top x+b<0\}$ would contain all $J$-points, hence $p$ too. That is a contradiction. So $d_{\mathrm{VC}}=d+1$, which for this class equals the number of parameters.

**4.** Write $c_j=1-y_j\in\{0,1\}$. Then

$$
\frac{\omega x_i}{\pi}=2^{-i}\Big(1+\sum_{j=1}^mc_j2^j\Big)=\underbrace{\sum_{j>i}c_j2^{j-i}}_{\text{even integer}}+c_i+\theta_i,\qquad \theta_i=\sum_{j<i}c_j2^{j-i}+2^{-i}.
$$

The even integer shifts $\omega x_i$ by a multiple of $2\pi$ and does not change the sine. Also $0<\theta_i\le\sum_{k=1}^{i-1}2^{-k}+2^{-i}=1-2^{-i}<1$. So $\omega x_i\equiv\pi(c_i+\theta_i)\pmod{2\pi}$ with $\theta_i\in(0,1)$, and $\sin(\omega x_i)>0$ iff $c_i=0$ iff $y_i=1$. Every labelling of every finite set $\{2^{-1},\dots,2^{-m}\}$ is produced, so $d_{\mathrm{VC}}=\infty$. No distribution-free VC bound applies to this one-parameter class.

The number of parameters is neither an upper bound on the VC dimension (this example; ReLU networks with $W$ weights in $L$ layers, whose VC dimension can be of order $WL\log(W/L)$) nor a lower bound (redundant parameters, as in $\mathbf 1[(w_1+w_2)x\ge0]$, add nothing).

If $\omega$ is a 32-bit float, the class has at most $2^{32}$ members. Shattering $m$ points needs $2^m$ distinct functions, so $d_{\mathrm{VC}}\le32$, and the finite-class bound applies with $\ln|\mathcal H|\le32\ln2$. The construction above needs $\omega/\pi$ to about $m+1$ significant bits. The infinite VC dimension comes from the infinite precision of one real number, which can hide arbitrarily many bits. This is the difference between counting parameters and counting bits that the float-encoding bound in 1.1 relies on.
"""
    ),
]

CELLS = REFERENCE + THEOREMS
