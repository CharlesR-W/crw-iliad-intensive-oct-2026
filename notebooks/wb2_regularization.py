"""MDL1 §4: regularization (ridge as prior, constraint and spectral filter; dropout; flatness).

Formulas are checked by checks/wb2a_check.py.
"""
from nbkit import *  # noqa: F401,F403

GBC7 = "https://www.deeplearningbook.org/contents/regularization.html"

CELLS = [
    markdown(
        r"""
<a id="wb2-section-4"></a>

## 4. Regularization: prior, constraint, filter, flatness

What weight decay, early stopping, dropout and flat minima each do to the
learned function, derived exactly in linear models, and why any claim about
flatness needs a metric.

Notation: $\lambda$ is the ridge penalty in $\frac1n\|y-X\beta\|^2+\lambda\|\beta\|^2$ (in 4.2 it is also the KKT multiplier, which turns out to be the same number), $a=n\lambda$, $s_k$ the singular values of $X$, $\sigma^2$ the noise variance, $\tau^2$ the prior variance, $\rho$ the squared constraint radius in 4.2, $t$ gradient-flow time, $q$ the dropout keep-probability, $H$ the Hessian.
"""
        + reading(
            "Sharp minima can generalize for deep nets",
            "https://arxiv.org/abs/1703.04933",
            "Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio (ICML 2017)",
            "Rescaling symmetries of ReLU networks make a minimum arbitrarily sharp without changing the "
            "function, so weight-space flatness alone cannot explain generalization. Read §2–§4; 4.5.3 is "
            "their construction in miniature.",
            importance=5, length=3,
        )
        + reading(
            "A continuous-time view of early stopping for least squares regression",
            "https://arxiv.org/abs/1810.10082",
            "Alnur Ali, J. Zico Kolter, Ryan J. Tibshirani (AISTATS 2019)",
            r"Compares the risks of gradient flow and ridge along the whole path, under the calibration "
            r"$t=1/\lambda$; the exact version of 4.3.3.",
            importance=2, length=3,
        )
        + reading(
            "Dropout training as adaptive regularization",
            "https://arxiv.org/abs/1307.1493",
            "Stefan Wager, Sida Wang, Percy Liang (NeurIPS 2013)",
            "Dropout in generalized linear models is, to second order, an L2 penalty in the metric of the "
            "diagonal Fisher information (4.4.3).",
            importance=2, length=2,
        )
        + r"""
Also (techniques not derived here): [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x), the lasso and sparsity; [Bishop (1995)](https://doi.org/10.1162/neco.1995.7.1.108), training with small input noise equals, to second order in the noise amplitude and for the expected loss, Tikhonov regularization by the input Jacobian (a Hessian term drops out at a minimum); [Chen, Dobriban and Lee (2020)](https://arxiv.org/abs/1907.10905), data augmentation as averaging over group orbits; [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101), decoupled weight decay (AdamW).
"""
        + references(
            book("Deep Learning", GBC7, "Goodfellow, Bengio and Courville (2016; free online)",
                 "ch. 7, a survey of every regularizer in common use"),
            book("The Elements of Statistical Learning", "https://hastie.su.domains/ElemStatLearn/",
                 "Hastie, Tibshirani and Friedman (2nd ed. 2009; free PDF)", "§3.4.1–3.4.3"),
        )
        + r"""
### Exercises: ridge as a prior, a constraint and a filter

Three readings of one penalty, which make different predictions about networks
trained on random labels (1.2.2).

**Setup for 4.1–4.3.** The linear model $y=X\beta+\varepsilon$ has design
$X\in\mathbb R^{n\times p}$, targets $y\in\mathbb R^n$, and noise $\varepsilon$
of mean $0$ and covariance $\sigma^2I_n$. Ridge regression with penalty
$\lambda>0$ is
$\hat\beta_\lambda=\arg\min_b\frac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y$.
The thin SVD is $X=\sum_ks_ku_kv_k^\top$, with singular values $s_k>0$ and
orthonormal $u_k\in\mathbb R^n$, $v_k\in\mathbb R^p$. $X^+$ is the
Moore–Penrose pseudoinverse, so $X^+y$ is the least-squares solution of
smallest norm.

**4.1. Ridge as a prior: MAP, posterior, and a warning.** *MAP* (maximum a
posteriori) is the mode of the posterior density. Take
$\varepsilon\sim\mathcal N(0,\sigma^2I_n)$.

1. Take the prior $\beta\sim\mathcal N(0,\tau^2I)$ and the likelihood $y\mid\beta\sim\mathcal N(X\beta,\sigma^2I)$. Show that the MAP estimate is ridge with $\lambda=\sigma^2/(n\tau^2)$. Show that the posterior is $\mathcal N\big(\hat\beta_\lambda,\ \sigma^2(X^\top X+n\lambda I)^{-1}\big)$, so the posterior mean equals the MAP here.
2. Compare the posterior covariance with the sampling covariance of $\hat\beta_\lambda$ over the noise at fixed $\beta$, which is $\sigma^2(X^\top X+n\lambda I)^{-1}X^\top X(X^\top X+n\lambda I)^{-1}$ (2.5). Which is larger, and why do they differ?
3. MAP depends on the parametrization. Suppose a posterior on $\beta>0$ has density $\pi(\beta)\propto\beta e^{-\beta}$. Find its mode. Now write $\beta=e^u$, find the mode of the density of $u$, and map it back. Which of the mode, mean and median is invariant under monotone reparametrization? What does this mean for "weight decay is a prior"?
"""
    ),
    answer(
        r"""
**1.** The negative log posterior is $\frac1{2\sigma^2}\|y-X\beta\|^2+\frac1{2\tau^2}\|\beta\|^2+\text{const}$. Multiplying by $2\sigma^2/n$ gives $\frac1n\|y-X\beta\|^2+\frac{\sigma^2}{n\tau^2}\|\beta\|^2$, the ridge objective with $\lambda=\sigma^2/(n\tau^2)$. The log posterior is quadratic in $\beta$, so the posterior is Gaussian. Its precision is $X^\top X/\sigma^2+I/\tau^2=(X^\top X+n\lambda I)/\sigma^2$, and its mean (which is also the mode) is $\hat\beta_\lambda$.

**2.** Write $G=X^\top X$ and $a=n\lambda$. The posterior covariance is $\sigma^2(G+aI)^{-1}$ and the sampling covariance is $\sigma^2G(G+aI)^{-2}$. Their difference is $\sigma^2a(G+aI)^{-2}\succeq0$, so the posterior is wider. The sampling covariance ignores the bias. The posterior covariance is the bias–variance mean squared error averaged over the prior: along a singular direction $v_k$ of $X$ with singular value $s$ (and $\mathbb E\beta_k^2=\tau^2$ under the prior), using 2.5.1 at $a=\sigma^2/\tau^2$,

$$
\frac{a^2\tau^2+\sigma^2s^2}{(s^2+a)^2}=\frac{\sigma^2(a+s^2)}{(s^2+a)^2}=\frac{\sigma^2}{s^2+a}.
$$

**3.** $\frac{d}{d\beta}\log(\beta e^{-\beta})=1/\beta-1=0$, so the mode is $\beta=1$. The density of $u=\log\beta$ is $\pi(e^u)e^u\propto e^{2u-e^u}$, and $\frac{d}{du}(2u-e^u)=0$ gives $e^u=2$, i.e. $\beta=2$. Same posterior, different "MAP". The mean is not invariant under nonlinear maps either. The median is invariant under monotone maps.

So the identification "penalty = negative log prior" holds *in a chosen parametrization*. Weight decay is a Gaussian prior on the weights, which in a network induces some prior on functions that depends on the architecture. The MAP estimate under it is not an invariant summary of the posterior.
"""
    ),
    markdown(
        r"""
**4.2. Ridge as a constraint: the Lagrange multiplier.** The second reading:
minimize the training loss over a ball $\|b\|^2\le \rho$. Small balls contain few
functions, which is the capacity explanation of why regularized models
generalize.

1. For $\rho>0$, show that a minimizer of $\hat R_n(b)=\frac1n\|y-Xb\|^2$ subject to $\|b\|^2\le \rho$ is $\hat\beta_\lambda$, where $\lambda$ is the KKT (Lagrange) multiplier. Interpret $\hat\beta_0=X^+y$ when the design is rank deficient.
2. Show that $\rho(\lambda)=\|\hat\beta_\lambda\|^2$ is strictly decreasing when $X^\top y\neq0$. Hence $\lambda(\rho)$ is well defined for $\rho<\|X^+y\|^2$, and $\lambda=0$ for $\rho\ge\|X^+y\|^2$.
3. In this view, what explains why the constrained estimator generalizes? Which objection from the random-label experiments (keystone) applies to that explanation?

<details><summary>Hint</summary>

For 1, write the Lagrangian $\hat R_n(b)+\lambda(\|b\|^2-\rho)$ with $\lambda\ge0$, set its gradient to zero, and use complementary slackness. For 2, write $\|\hat\beta_\lambda\|^2$ in the SVD of $X$ and look at each term as a function of $\lambda$. For 3, recall 2.1.3; the Rademacher complexity of a norm ball of linear predictors is in *Understanding Machine Learning* §26.2.
</details>
"""
    ),
    answer(
        r"""
**1.** The problem is convex, and Slater's condition holds for $\rho>0$. The Lagrangian is $\hat R_n(b)+\lambda(\|b\|^2-\rho)$ with $\lambda\ge0$. Stationarity gives $-\frac2nX^\top(y-Xb)+2\lambda b=0$, i.e. $b=(X^\top X+n\lambda I)^{-1}X^\top y=\hat\beta_\lambda$ for $\lambda>0$. For $\lambda=0$, use the minimum-norm least-squares solution $X^+y$; other feasible least-squares solutions may also minimize when $X$ is rank deficient. Complementary slackness says either $\lambda=0$ (the constraint need not be strictly slack) or $\|\hat\beta_\lambda\|^2=\rho$.

**2.** In the SVD, $\|\hat\beta_\lambda\|^2=\sum_k\frac{s_k^2(u_k^\top y)^2}{(s_k^2+n\lambda)^2}$. Each term is non-increasing in $\lambda$, and at least one is strictly decreasing when $X^\top y\ne0$. As $\lambda\downarrow0$ the norm rises to $\|X^+y\|^2$, the minimum-norm least-squares solution, and as $\lambda\to\infty$ it falls to $0$. So the map from $\rho$ to $\lambda$ is a decreasing bijection on $(0,\|X^+y\|^2)$. Tighter constraints need bigger multipliers.

**3.** The explanation is capacity. The norm ball $\{x\mapsto x^\top b:\|b\|\le B\}$ has Rademacher complexity of order $B\sqrt{\mathbb E\|x\|^2/n}$ (quoted), which controls prediction complexity. To turn this into a squared-loss gap bound, additionally control the input norms and labels (or suitable tails), so the loss is Lipschitz on the allowed prediction range. Small enough complexity then gives an excess-risk bound via 2.1.3. This inherits two problems. The norm depends on the parametrization (4.1.3, 4.5), and Zhang et al.'s objection applies: nets with weight decay still fit random labels, so the constraint that is actually active is not small.
"""
    ),
    markdown(
        r"""
<a id="wb2-spectral-filter"></a>

**4.3. Ridge as a spectral filter, and early stopping.** The third reading:
ridge acts on the principal directions $v_k$ one at a time, keeping the
well-determined ones (large $s_k$) and shrinking the rest. Gradient descent
stopped early does something similar. A *filter* is a set of factors
$\phi_k\in[0,1]$ saying how much of each direction an estimator keeps.

1. Show that $\hat\beta_\lambda=\sum_{k:\,s_k>0}\phi_k\,\dfrac{u_k^\top y}{s_k}\,v_k$ with $\phi_k^{\mathrm{ridge}}=\dfrac{s_k^2}{s_k^2+n\lambda}$. The factor $(u_k^\top y)/s_k$ is the unregularized coefficient.
2. Run gradient flow $\dot\beta=-\nabla L(\beta)$ on $L(\beta)=\frac1{2n}\|X\beta-y\|^2$, starting from $\beta(0)=0$. Show that $\beta(t)$ has the same form with $\phi_k^{\mathrm{GF}}(t)=1-e^{-s_k^2t/n}$. For gradient descent with step $\eta$ and $T$ steps, show $\phi_k=1-(1-\eta s_k^2/n)^T$.
3. Show that the two filters cross $\tfrac12$ at the same $s^2$ if and only if $t=\ln2/\lambda$. With that matching, write both in terms of $x=s^2/(n\lambda)$, and compare them for $x\ll1$ and for $x\gg1$. Which filter passes more of the weak directions? Which shrinks the strong directions more?
4. Let $p>n$. Show that gradient flow from $\beta(0)=0$ converges to the minimum-norm interpolant $X^+y$. What is the limit from a nonzero initialization $\beta_0$?

<details><summary>Hint</summary>

Work in the coordinates $b_k=v_k^\top\beta$, where $X^\top X$ is diagonal and $X^\top y=\sum_ks_k(u_k^\top y)v_k$; gradient flow then decouples into one scalar linear ODE per $k$. For 3, set each filter equal to $\tfrac12$ and solve for $s^2$. For 4, in which subspace does $\dot\beta$ always lie?
</details>
"""
    ),
    answer(
        r"""
**1.** $X^\top y=\sum_k s_k(u_k^\top y)v_k$ and $(X^\top X+n\lambda I)^{-1}v_k=v_k/(s_k^2+n\lambda)$, so $\hat\beta_\lambda=\sum_k\frac{s_k(u_k^\top y)}{s_k^2+n\lambda}v_k=\sum_k\frac{s_k^2}{s_k^2+n\lambda}\cdot\frac{u_k^\top y}{s_k}v_k$.

**2.** $\nabla L=\frac1n(X^\top X\beta-X^\top y)$. In coordinates $b_k=v_k^\top\beta$,

$$
\dot b_k=-\frac{s_k^2}{n}b_k+\frac{s_k}{n}u_k^\top y,
$$

with fixed point $(u_k^\top y)/s_k$. From $b_k(0)=0$ the solution is $b_k(t)=(1-e^{-s_k^2t/n})(u_k^\top y)/s_k$. For gradient descent, $b_k^{(T)}-b_k^\infty=(1-\eta s_k^2/n)^T(b_k^{(0)}-b_k^\infty)$, which gives the stated filter. Directions with $s_k=0$ never move. Discrete gradient descent converges on the positive eigendirections when $0<\eta<2n/s_{\max}^2$; the filter is non-oscillatory for $\eta\le n/s_{\max}^2$.

**3.** Ridge equals $\tfrac12$ at $s^2=n\lambda$, and flow equals $\tfrac12$ at $s^2t/n=\ln2$, i.e. at $s^2=n\ln2/t$. These coincide iff $t=\ln2/\lambda$. With this matching, ridge is $x/(1+x)$ and flow is $1-e^{-x\ln2}=1-2^{-x}$.
- *Weak directions,* $x\ll1$: ridge $\approx x$ and flow $\approx x\ln2$, so ridge passes about $1/\ln2\approx1.44$ times more of each weak direction.
- *Strong directions,* $x\gg1$: ridge leaves a shrinkage of $1-\phi\approx1/x$, only polynomially small, while flow leaves $2^{-x}$, exponentially small. So ridge shrinks the strong directions more.

Gradient flow is the sharper, more step-like filter. The largest disagreements are flow above ridge by about $0.138$ (near $x\approx4.3$) and ridge above flow by about $0.044$ (near $x\approx0.36$). With the naive matching $t=1/\lambda$, flow becomes $1-e^{-x}$ and its half-point moves to $x=\ln2$. "Early stopping is ridge" is right in outline (keep the strong directions, drop the weak ones) and wrong in detail. MDL4 (learning curves) derives how the polynomial tail of the ridge filter affects learning-curve exponents.

**4.** $\dot\beta\in\mathrm{row}(X)$, so $\beta(t)\in\mathrm{row}(X)$ for all $t$. For $p>n$ with $X$ of full row rank, all $s_k>0$ for $k\le n$, so by 2 the flow converges to $\sum_{k\le n}\frac{u_k^\top y}{s_k}v_k=X^+y$. That is the unique interpolant in the row space, and so the minimum-norm interpolant. From $\beta_0$, the row-space component converges to the same place and the null-space component never moves: the limit is $X^+y+(I-X^+X)\beta_0$. The implicit bias is "minimum distance from the initialization".
"""
    ),
    markdown(
        figure(
            "assets/interactive/spectral_filters.html",
            "Use panel A only (panel B belongs to MDL4). Press “match half-points” and read the difference "
            "strip: where do the filters disagree, and which passes more of the weak tail? Then set "
            "t = 1/λ by hand and see the ln 2 offset.",
        )
    ),
    markdown(
        r"""
<a id="wb2-dropout"></a>

### Exercises: dropout

**4.4. Dropout as adaptive $L^2$.** In a linear model, averaging over the
dropout noise gives an explicit penalty. Take a linear predictor
$f(x)=\beta^\top x$, training pairs $(x_i,y_i)$ with $x_i\in\mathbb R^p$ the
rows of $X\in\mathbb R^{n\times p}$, and squared loss. Drop inputs
independently: $\tilde x_{ij}=x_{ij}\xi_{ij}/q$ with
$\xi_{ij}\sim\mathrm{Bernoulli}(q)$, where $q$ is the keep probability; the
$1/q$ keeps $\mathbb E\tilde x_{ij}=x_{ij}$.

1. Show that $\mathbb E_\xi\frac1n\sum_i(y_i-\tilde x_i^\top\beta)^2=\frac1n\|y-X\beta\|^2+\frac{1-q}{q}\sum_j\beta_j^2\,\frac{\|x_{\cdot j}\|^2}{n}$, where $x_{\cdot j}$ is the $j$-th column of $X$.
2. Write the penalty as $\beta^\top\Gamma\beta$. Show that it is invariant under rescaling a feature ($x_{\cdot j}\to \alpha x_{\cdot j}$, $\beta_j\to\beta_j/\alpha$), while the ridge penalty is not. Which standard procedure is it equivalent to?
3. Take a generalized linear model with per-example loss $A(x^\top\beta)-y\,x^\top\beta$ (for logistic regression, $A(\eta)=\log(1+e^\eta)$). Expand to second order in the dropout noise and show that the expected penalty is approximately $\frac{1-q}{2q}\cdot\frac1n\sum_iA''(x_i^\top\beta)\sum_jx_{ij}^2\beta_j^2$. In what sense is this "adaptive"?

<details><summary>Hint</summary>

Write $\tilde x_i^\top\beta=x_i^\top\beta+\sum_jx_{ij}\beta_j(\xi_{ij}/q-1)$. The noise has mean zero and $\mathrm{Var}(\xi/q)=(1-q)/q$.
</details>
"""
    ),
    answer(
        r"""
**1.** Let $\eta_i=x_i^\top\beta$ and $\delta_i=\tilde x_i^\top\beta-\eta_i=\sum_jx_{ij}\beta_j(\xi_{ij}/q-1)$. Then $\mathbb E\delta_i=0$ and, by independence, $\mathrm{Var}\,\delta_i=\frac{1-q}{q}\sum_jx_{ij}^2\beta_j^2$. So $\mathbb E(y_i-\eta_i-\delta_i)^2=(y_i-\eta_i)^2+\mathrm{Var}\,\delta_i$. Averaging over $i$ gives the penalty $\frac{1-q}{q}\sum_j\beta_j^2\|x_{\cdot j}\|^2/n$.

**2.** $\Gamma=\frac{1-q}{q}\,\mathrm{diag}(X^\top X/n)$. Under $x_{\cdot j}\to \alpha x_{\cdot j}$, $\beta_j\to\beta_j/\alpha$, the product $\beta_j^2\|x_{\cdot j}\|^2$ is unchanged, while ridge's $\beta_j^2$ changes by $1/\alpha^2$. Dropout on inputs is ridge after standardizing each feature by its root mean square: ridge in the metric $\mathrm{diag}(\hat\Sigma)$.

**3.** $\mathbb E\tilde x_i^\top\beta=\eta_i$, so the linear term $-y\,\tilde x_i^\top\beta$ is unchanged in expectation. A second-order expansion gives $\mathbb E A(\eta_i+\delta_i)\approx A(\eta_i)+\tfrac12A''(\eta_i)\mathrm{Var}\,\delta_i$, so the penalty is $\frac1n\sum_i\tfrac12A''(\eta_i)\frac{1-q}{q}\sum_jx_{ij}^2\beta_j^2$. For logistic regression $A''(\eta)=\hat p(1-\hat p)$, the Fisher information of the example. The penalty is ridge in the metric of the diagonal Fisher information. Confidently classified examples contribute little, and features that matter on uncertain examples are penalized most. "Adaptive" because the metric depends on the current $\beta$. For squared loss written as $\tfrac12(y-\eta)^2$, $A''=1$, which recovers 1 up to the factor $\tfrac12$ of that convention.

The general lesson: many "hacky" regularizers are quadratic penalties in disguise, in some metric. The question to ask is "which metric?"
"""
    ),
    markdown(
        r"""
<a id="wb2-flatness"></a>

### Exercises: flatness

The exact local statement behind "flat minima generalize", and why the naive
version cannot be right.

**4.5. Why flatness could matter, and why sharpness needs a metric.** Near a
minimum $\theta^*$ of the population risk $R$, write $H=\nabla^2R(\theta^*)$
for the Hessian (large eigenvalues are *sharp* directions, small ones *flat*).
A trained parameter $\hat\theta=\theta^*+\delta$ is random through the data
and the optimizer; write $\Sigma=\mathbb E\,\delta\delta^\top$ for its spread.
The *per-sample gradient covariance* at the optimum is
$C=\mathrm{Cov}_z[\nabla_\theta\ell(\theta^*;z)]$, for a loss $\ell(\theta;z)$
on one example $z$.

1. Show that $\mathbb E R(\hat\theta)-R(\theta^*)\approx\tfrac12\mathrm{Tr}(H\Sigma)$. Check that this is exact for least squares with test inputs $x\sim\mathcal N(0,I_p)$, where $R(b)=\sigma^2+\|b-\beta\|^2$ and $\Sigma=\sigma^2(X^\top X)^{-1}$ (2.2).
2. Suppose $\hat\theta$ is ERM in a regular model with $\Sigma\approx\frac1nH^{-1}CH^{-1}$ (the "sandwich" covariance, derived in 7.1). Show that $\tfrac12\mathrm{Tr}(H\Sigma)=\frac1{2n}\mathrm{Tr}(H^{-1}C)$, and that if $C=cH$ this equals $cp/(2n)$, whatever the curvature. What has to be true of $\Sigma$ for "flatter is better" to hold?
3. *Dinh et al.'s rescaling.* Take the two-layer scalar linear network $f(x)=w_2w_1x$ with $R(w)=\mathbb E(y-w_2w_1x)^2$ and $a=\mathbb E x^2>0$. Assume $c^*=\mathbb E[xy]/a\ne0$; the minima are the hyperbola $w_1w_2=c^*$. Compute the Hessian at a minimum, show that $\mathrm{Tr}\,H=2a(w_1^2+w_2^2)$ and that one eigenvalue is zero. Under $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ the function does not change. Show that $\mathrm{Tr}\,H$ takes every value in $[4a|c^*|,\infty)$. Why does the same trick work in deep ReLU networks?
4. Let $\theta=g(\varphi)$ be a smooth reparametrization with Jacobian $J$ at the minimum, and let $\delta$ transform as a tangent vector. Show that $\mathrm{Tr}(H\Sigma)$ is invariant while $\mathrm{Tr}\,H$ is not. What does a meaningful flatness claim have to specify?

<details><summary>Hint</summary>

For 1, the gradient vanishes at $\theta^*$, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\,\mathbb E\,\delta\delta^\top)$. For 3, write $R=a(w_1w_2-c^*)^2+\text{const}$. For 4, at a critical point the Hessian transforms as $J^\top HJ$, because the term involving the second derivative of $g$ multiplies $\nabla R=0$.
</details>
"""
    ),
    answer(
        r"""
**1.** $R(\theta^*+\delta)=R(\theta^*)+\nabla R(\theta^*)^\top\delta+\tfrac12\delta^\top H\delta+O(\|\delta\|^3)$. The gradient term vanishes, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\Sigma)$. Here $\Sigma$ is the second moment about $\theta^*$, so it includes any bias. For OLS, $R(b)=\sigma^2+\|b-\beta\|^2$ is exactly quadratic with $H=2I$, and $\Sigma=\sigma^2(X^\top X)^{-1}$, so $\tfrac12\mathrm{Tr}(H\Sigma)=\sigma^2\mathrm{Tr}(X^\top X)^{-1}$ exactly, as in 2.2.2.

**2.** $\tfrac12\mathrm{Tr}\big(H\cdot\tfrac1nH^{-1}CH^{-1}\big)=\tfrac1{2n}\mathrm{Tr}(CH^{-1})$. If $C=cH$, this is $cp/(2n)$. The curvature cancels: a sharp direction is pinned down more tightly by the data ($\Sigma\propto H^{-1}$), and this exactly offsets the larger cost of a given error there. "Flatter is better" needs $\Sigma$ that does not shrink in proportion as $H$ grows. One case is a perturbation scale set by something other than the data's curvature, such as SGD noise at a fixed learning rate: if the gradient noise scales as $C\propto H^{g}$, 8.2 finds $\Sigma\propto H^{g-1}$ (it writes the exponent as $\gamma$). So "flat minima generalize" is a claim about $H$ *and* $\Sigma$ jointly.

**3.** Write $u=w_1w_2$, so $R=a(u-c^*)^2+\text{const}$. Then $\partial_{w_1}R=2a(u-c^*)w_2$, and at a minimum ($u=c^*$),

$$
H=2a\begin{pmatrix}w_2^2&w_1w_2\\ w_1w_2&w_1^2\end{pmatrix}=2a\,vv^\top,\qquad v=(w_2,w_1)^\top .
$$

The terms proportional to $u-c^*$ vanish there. So $H$ has rank one, with eigenvalue $2a(w_1^2+w_2^2)=\mathrm{Tr}\,H$ along $v$ and eigenvalue $0$ along $(w_1,-w_2)$, the direction tangent to the hyperbola. Under the rescaling, $\mathrm{Tr}\,H=2a(\alpha^2w_1^2+w_2^2/\alpha^2)\ge4a|w_1w_2|=4a|c^*|$ by AM–GM, with equality at $\alpha^2=|w_2/w_1|$, and it is unbounded as $\alpha\to0$ or $\infty$. Every value in $[4a|c^*|,\infty)$ is attained by the same function. Weight decay would pick the balanced representative. For ReLU networks, $\mathrm{ReLU}(\alpha z)=\alpha\,\mathrm{ReLU}(z)$ for $\alpha>0$, so multiplying one layer by $\alpha$ and the next by $1/\alpha$ leaves the network function unchanged. Dinh et al. use this to make any minimum arbitrarily sharp.

**4.** At a critical point, $H_\varphi=J^\top HJ$. To first order $\delta_\theta=J\delta_\varphi$, so $\Sigma_\varphi=J^{-1}\Sigma J^{-\top}$. Then $\mathrm{Tr}(H_\varphi\Sigma_\varphi)=\mathrm{Tr}(J^\top HJJ^{-1}\Sigma J^{-\top})=\mathrm{Tr}(H\Sigma)$, which is invariant, whereas $\mathrm{Tr}\,H_\varphi=\mathrm{Tr}(J^\top HJ)$ depends on $J$. A meaningful flatness claim has to say *in what metric* the perturbations are measured: either the distribution $\Sigma$ of perturbations that actually occurs, or an invariant geometry such as the Fisher (output-space) metric. Raw $\mathrm{Tr}\,H$ in weight space is a property of the parametrization, not of the function.
"""
    ),
    markdown(
        r"""
**4.6. What Dinh et al. show.** Dinh et al. (reading above) is often cited as
"flatness is meaningless". Using 4.5.4, say precisely what the paper shows and
what it leaves open. Name one flatness-like quantity it does not touch.
"""
    ),
    answer(
        r"""
The paper shows that weight-space sharpness measures ($\mathrm{Tr}\,H$, the top eigenvalue, $\epsilon$-sharpness in a Euclidean ball, flat volume) are not functions of the network function, because a symmetry of the parametrization changes them arbitrarily. So none of them can be a sufficient statistic for generalization, and an empirical correlation between them and test error depends on how parametrizations happen to be distributed in practice. It leaves open quantities that are invariant under the symmetry or carry their own metric:
- $\mathrm{Tr}(H\Sigma)$ with the $\Sigma$ that training actually produces;
- sharpness measured in the Fisher or output-space metric;
- normalized or scale-invariant sharpness;
- the volume of a function-space neighbourhood (the posterior mass of a function), which is what PAC-Bayes and Bayesian accounts use.

It also leaves open whether SGD, from standard initializations, tends to land in parametrizations where raw sharpness *happens* to track one of these. That is an empirical question about the training procedure, not about the geometry.
"""
    ),
]
