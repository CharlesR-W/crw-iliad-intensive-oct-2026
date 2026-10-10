"""WB2 §4: Bayes, MLE, curvature and generalization measures. Checked by checks/wb2b_check.py.
"""
from nbkit import *  # noqa: F401,F403

S4 = [
    # ------------------------------------------------------------------ §4.A
    markdown(
        r"""
<a id="wb2-section-4"></a>

## 4. Bayes, MLE, and generalization measures

**The story.** Many claims about why networks generalize ("count the
parameters", "flat minima generalize", "SGD is approximately Bayesian", "the
Bayesian posterior is what matters") are claims about one classical formula:
how much worse a fitted model does on new data than the best model, to leading
order in $1/n$. This section derives that formula and follows it until it
breaks. It is one story in three parts.

- **4.A. Bayes and MLE generalization error.** When there are many more data
  than parameters, the error of the fitted parameters is approximately
  Gaussian, and the expected excess risk is $\mathrm{Tr}(H^{-1}C)/2n$, built
  from a curvature matrix $H$ and a gradient-noise matrix $C$ (defined below).
  For a correctly specified probabilistic model $C=H$, curvature cancels, and
  the answer is parameter counting, $p/2n$: this is the AIC model-selection
  criterion. The Bayesian side gives BIC, and a comparison of three different
  things called "the generalization error" (a single fit, the Bayes ensemble,
  a single posterior draw). The terms are defined just below.
- **4.B. Does curvature cancel?** For trained networks $C$ is not proportional to $H$. How the
  two differ, how SGD's own noise spreads the parameters around a minimum, and
  whether that spread looks like a Bayesian posterior (it does not, in the
  measured regime).
- **4.C. Fantastic generalization measures.** A large empirical study of which
  proposed measures actually predict generalization in networks, read against
  4.A and 4.B.

**The two matrices.** Let $\ell(\theta;z)$ be the loss of parameters
$\theta\in\mathbb R^p$ on one example $z=(x,y)$, $R(\theta)=\mathbb E_z\ell(\theta;z)$
the population risk (test loss), $\hat R_n(\theta)=\frac1n\sum_{i=1}^n\ell(\theta;z_i)$
the empirical risk (training loss) on $n$ i.i.d. examples, and $\theta^*$ the
minimizer of $R$ (as in 1.A).

- The **curvature** $H=\nabla^2R(\theta^*)$ is the Hessian of the population
  loss at the optimum. Its eigenvalues say how fast the loss rises as you move
  away from $\theta^*$ along each eigenvector: large eigenvalues are *sharp*
  directions, small ones *flat*.
- The **gradient noise** $C=\mathrm{Cov}_z\big[\nabla_\theta\ell(\theta^*;z)\big]$
  is the covariance of single-example gradients at the optimum. The average
  gradient there is zero, but individual examples still pull in different
  directions, and $C$ measures how much. Since the gradient of the training
  loss at $\theta^*$ is an average of $n$ such terms, it is approximately
  $\mathcal N(0,C/n)$: this is what makes the fitted parameters wobble from one
  training set to another.

The link between them and the test loss is the local formula of 1.D.7.1 (a
second-order Taylor expansion of $R$ around $\theta^*$): if a
trained parameter is $\theta=\theta^*+\delta$ with spread
$\Sigma=\mathbb E\,\delta\delta^\top$, then, because $\nabla R(\theta^*)=0$,

$$
\mathbb E\,R(\theta)-R(\theta^*)\approx\tfrac12\mathrm{Tr}(H\Sigma).
$$

Different learners (ERM, a posterior sample, SGD) differ in their $\Sigma$, and
that is the whole content of 4.A–4.B.

**Words used in this section.**

- *Log loss, likelihood, MLE.* If the model outputs a probability
  $p_\theta(y\mid x)$, the log loss is $\ell=-\log p_\theta(y\mid x)$, in nats.
  The log-likelihood of the training set is $\log L(\theta)=-n\hat R_n(\theta)$,
  so ERM with the log loss is maximum likelihood estimation (MLE). The model is
  *well specified* if the data really come from $p_{\theta^*}$ for some
  $\theta^*$, and *misspecified* otherwise.
- *Model-selection criteria* score each candidate model by its fitted training
  loss plus a complexity penalty; you pick the model with the lowest score.
  **AIC** (Akaike's information criterion) is $-2\log L(\hat\theta_n)+2p$; up
  to a factor $2n$, it estimates the fitted model's test log loss, assuming the
  model is well specified (4.A.4). **TIC** (Takeuchi's information criterion)
  replaces $p$ by $\mathrm{Tr}(H^{-1}C)$ and drops that assumption (4.A.2).
  **BIC** (the Bayesian, or Schwarz, information criterion) is
  $-2\log L(\hat\theta_n)+p\log n$; it approximates $-2\log$ of the Bayesian
  evidence and aims to pick the model most likely to be true (4.A.5).
- *Posterior, posterior predictive, temperature.* With a prior density
  $\pi(\theta)$ and the log loss, the Bayesian posterior is
  $\pi_n(\theta)\propto\pi(\theta)\,e^{-n\hat R_n(\theta)}$ (prior times
  likelihood). The *posterior predictive* $\bar p_n(y\mid x)=\int
  p_\theta(y\mid x)\,\pi_n(d\theta)$ averages the predictions over the
  posterior: the "Bayes ensemble". A *tempered posterior* at temperature $T>0$
  is $\pi_T(\theta)\propto\pi(\theta)\,e^{-n\hat R_n(\theta)/T}$. $T=1$ is
  ordinary Bayes; $T>1$ widens the posterior, as if there were only $n/T$
  data points, and $T<1$ narrows it. The name comes from statistical physics,
  where $e^{-E/T}$ is the Boltzmann distribution of energy $E$ at temperature
  $T$; here the energy is $n\hat R_n$.

<a id="wb2-section-4a"></a>

### 4.A. Bayes and MLE generalization error

**The question.** In the classical regime, how large are the excess risk and
the gap between test and training loss of the empirical risk minimizer, and
how do these compare with what a Bayesian would predict? The answers are the
model-selection criteria AIC, TIC and BIC, and the exercises derive each one.

**The regular regime.** All results in 4.A assume:

- the number of parameters $p$ is fixed and $n\to\infty$, and $\theta^*$ is the unique minimizer of $R$, in the interior of the parameter space;
- $H\succ0$ (no flat directions at the optimum), and $\hat\theta_n\to\theta^*$ in probability;
- enough smoothness and moments for Taylor expansions, the law of large numbers and the central limit theorem.

A model satisfying these is *regular*. Neural networks are not: they have
many more parameters than the regime allows, and their Hessians have flat
directions from symmetries. Models whose $H$ is degenerate are called
*singular*; singular learning theory (SLT) is the extension to them, and it
appears in 4.A.5–4.A.6 only as a pointer. The value of the regular theory is as
an exact reference case.

**Notation.** The population risk is $R(\theta)=\mathbb E_z\ell(\theta;z)$,
with minimizer $\theta^*$, and ERM is $\hat\theta_n=\arg\min\hat R_n$, where
$\hat R_n(\theta)=\frac1n\sum_i\ell(\theta;z_i)$. The curvature is
$H=\nabla^2R(\theta^*)$ and the gradient noise
$C=\mathrm{Cov}_z[\nabla\ell(\theta^*;z)]$. Because
$\mathbb E\nabla\ell(\theta^*;z)=\nabla R(\theta^*)=0$, the gradient noise is
also the second moment,
$$
C=\mathbb E\big[\nabla\ell(\theta^*;z)\nabla\ell(\theta^*;z)^\top\big].
$$
The *optimism* of ERM is the expected generalization gap
$\mathbb E[R(\hat\theta_n)-\hat R_n(\hat\theta_n)]$ (1.A's $\mathbb E\,\mathrm{gap}_n$):
how much the training loss flatters the fitted model. The *excess risk* is
$\mathbb E\,R(\hat\theta_n)-R(\theta^*)$: how much worse the fitted model is
than the best one in the model.

*Sources:* the sandwich is due to Huber (1967) and, for misspecified
likelihoods, H. White, "Maximum likelihood estimation of misspecified models",
*Econometrica* (1982). AIC is from Akaike (1974) and BIC from
[Schwarz (1978)](https://doi.org/10.1214/aos/1176344136); [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (ESL) §7.4–7.5 and
§7.7 derive both in textbook form.

**Loss conventions.** All excess risks below are in loss units. For squared
loss this section uses $\ell=\tfrac12(y-f_\theta(x))^2$. Its Hessian is
$\nabla^2\ell=\nabla f\,\nabla f^\top-(y-f)\nabla^2f$. The first piece is the
*Gauss–Newton* term (with no factor of 2 in this convention); the second, a
residual times a second derivative, has zero population expectation when the
model's mean $f_{\theta^*}(x)=\mathbb E[y\mid x]$ is correct. (1.B used $(y-f)^2$, which doubles $H$
and quadruples $C$; $\mathrm{Tr}(H^{-1}C)$ doubles.) For the log loss
$\ell=-\log p_\theta(y\mid x)$, in nats, and a well-specified model,
$R(\theta)-R(\theta^*)=\mathbb E_x\,\mathrm{KL}\big(p_{\theta^*}(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x)\big)$,
so excess risk is the KL divergence to the truth. (Recall
$\mathrm{KL}(q\,\|\,p)=\mathbb E_{y\sim q}\log\frac{q(y)}{p(y)}\ge0$, with equality
iff $p=q$: the expected extra log loss from predicting with $p$ when the data
follow $q$.)

#### 4.A.1. The sandwich: asymptotic normality of ERM

**Why.** Everything in 4.A follows from the distribution of the fitted
parameters. The result is a Gaussian whose covariance is "sandwiched" between
two inverse curvatures, with the gradient noise as the filling.

Starting from the first-order condition $\nabla\hat R_n(\hat\theta_n)=0$, show
that
$$
\sqrt n\,(\hat\theta_n-\theta^*)\ \Rightarrow\ \mathcal N\big(0,\ H^{-1}CH^{-1}\big).
$$
Here $\Rightarrow$ means convergence in distribution. Classical maximum-likelihood
theory instead says that the MLE's covariance is approximately $I^{-1}/n$, where the *Fisher
information* $I=\mathbb E\big[\nabla\log p_\theta\,\nabla\log p_\theta^\top\big]=-\mathbb E\,\nabla^2\log p_\theta$
(the two expressions agree for a well-specified model; 4.A.4). Which of $H$ and
$C$ does that formula use, and when does it give the wrong answer?

<details><summary>Hint</summary>

Taylor-expand $\nabla\hat R_n$ around $\theta^*$. Apply the law of large numbers to $\nabla^2\hat R_n(\theta^*)$ and the central limit theorem to $\sqrt n\,\nabla\hat R_n(\theta^*)$, then Slutsky's lemma (if $A_n\to A$ in probability and $b_n\Rightarrow b$, then $A_nb_n\Rightarrow Ab$).
</details>
"""
    ),
    answer(
        r"""
Expanding,
$0=\nabla\hat R_n(\hat\theta_n)=\nabla\hat R_n(\theta^*)+\nabla^2\hat R_n(\tilde\theta)\,(\hat\theta_n-\theta^*)$
for some $\tilde\theta$ between $\hat\theta_n$ and $\theta^*$ (row by row). So
$$
\sqrt n(\hat\theta_n-\theta^*)=-\big[\nabla^2\hat R_n(\tilde\theta)\big]^{-1}\sqrt n\,\nabla\hat R_n(\theta^*).
$$
$\nabla\hat R_n(\theta^*)=\frac1n\sum_i\nabla\ell(\theta^*;z_i)$ is an average
of i.i.d. vectors with mean $0$ and covariance $C$, so by the CLT
$\sqrt n\,\nabla\hat R_n(\theta^*)\Rightarrow\mathcal N(0,C)$. By the LLN and
consistency, $\nabla^2\hat R_n(\tilde\theta)\to H$ in probability. By
Slutsky, the limit is $-H^{-1}\mathcal N(0,C)=\mathcal N(0,H^{-1}CH^{-1})$.
This is the *sandwich*, with $H^{-1}$ as the bread and $C$ as the filling
(Huber; White 1982 for misspecified likelihoods).

The "inverse Fisher information" formula is $H^{-1}$ (or $C^{-1}$; for the
log loss, $C$ is the score form of $I$ and $H$ the Hessian form), which
equals the sandwich only when $C=H$ (4.A.4). Misspecification can break this equality, although it need not do so
in every model. Example: heteroscedastic noise fitted with a homoscedastic
model, as in 4.A.3.
"""
    ),
    markdown(
        r"""
#### 4.A.2. Excess risk, optimism, and a factor of two

**Why.** Two different numbers summarize how ERM generalizes: its excess risk
over the best parameter, and its optimism (how much the training loss flatters
it). Both follow from the sandwich, and they differ by exactly a factor of two.

1. Show that $\mathbb E\,R(\hat\theta_n)-R(\theta^*)\approx\frac1{2n}\mathrm{Tr}(H^{-1}C)$.
2. Show that the expected optimism is twice as large:
$$
\mathbb E\big[R(\hat\theta_n)-\hat R_n(\hat\theta_n)\big]\approx\frac1n\mathrm{Tr}(H^{-1}C).
$$
This is Takeuchi's information criterion (TIC): estimate the test loss by $\hat R_n(\hat\theta_n)+\frac1n\mathrm{Tr}(\hat H^{-1}\hat C)$, where $\hat H$ and $\hat C$ are the training-set averages of $\nabla^2\ell$ and of $\nabla\ell\,\nabla\ell^\top$ at $\hat\theta_n$.
3. Explain the factor of two in one sentence. Which of the two quantities does a held-out set estimate?

<details><summary>Hint</summary>

For 2, expand $\hat R_n$ around its own minimizer $\hat\theta_n$ and evaluate the expansion at $\theta^*$. Then use $\mathbb E\hat R_n(\theta^*)=R(\theta^*)$.
</details>

#### 4.A.3. A concrete misspecified case

**Why.** When the model is wrong about the noise, "count the parameters"
can be off by a large factor; here is a one-parameter example where it is
off by 3.

Let $x\sim\mathcal N(0,1)$ and $y=\beta x+\varepsilon$, where
$\varepsilon\mid x\sim\mathcal N(0,\tau^2x^2)$. The noise is *heteroscedastic*: its variance depends on $x$.
Fit $y\approx\theta x$ with $\ell=\tfrac12(y-\theta x)^2$.

1. Compute $H$, $C$, the asymptotic variance of $\hat\theta_n$, the expected excess risk and the expected optimism.
2. Compare each with what you get by assuming homoscedastic noise of the same average variance, $\mathbb E\varepsilon^2=\tau^2$, i.e. with $C=\tau^2H$. By what factor is the naive answer wrong?

<details><summary>Hint</summary>

At $\theta^*=\beta$ the per-sample gradient is $-\varepsilon x$. For a standard normal, $\mathbb E x^4=3$.
</details>
"""
    ),
    answer(
        r"""
**4.A.2.1.** By 4.A.1, $\delta=\hat\theta_n-\theta^*$ has
$\mathbb E\,\delta\delta^\top\approx\Sigma_n:=H^{-1}CH^{-1}/n$. By the 1.D.7
formula, the expected excess risk is
$\tfrac12\mathrm{Tr}(H\Sigma_n)=\frac1{2n}\mathrm{Tr}(HH^{-1}CH^{-1})=\frac1{2n}\mathrm{Tr}(H^{-1}C)$.

**4.A.2.2.** Expand $\hat R_n$ around its minimizer, where its gradient
vanishes:
$\hat R_n(\theta^*)\approx\hat R_n(\hat\theta_n)+\tfrac12\delta^\top\hat H\delta$,
with $\hat H=\nabla^2\hat R_n\approx H$. Take expectations, using
$\mathbb E\hat R_n(\theta^*)=R(\theta^*)$ because $\theta^*$ does not depend
on the sample:
$$
\mathbb E\hat R_n(\hat\theta_n)\approx R(\theta^*)-\tfrac1{2n}\mathrm{Tr}(H^{-1}C).
$$
Subtracting this from 1 gives
$\mathbb E[R(\hat\theta_n)-\hat R_n(\hat\theta_n)]\approx\big(\tfrac1{2n}+\tfrac1{2n}\big)\mathrm{Tr}(H^{-1}C)=\frac1n\mathrm{Tr}(H^{-1}C)$.

**4.A.2.3.** The test loss of $\hat\theta_n$ is above $R(\theta^*)$ by
$\mathrm{Tr}(H^{-1}C)/2n$, and its training loss is *below* $R(\theta^*)$ by
the same amount, because $\hat\theta_n$ was fitted to that sample's noise. The
optimism spans both. A held-out set estimates $R(\hat\theta_n)$, so held-out
minus training loss estimates the optimism. Held-out loss cannot give the
excess risk without the unknown $R(\theta^*)$, although differences of excess
risk between two models are measurable.

**4.A.3.1.** $\nabla\ell=-(y-\theta x)x$, so $H=\mathbb E x^2=1$. At
$\theta^*=\beta$, $\nabla\ell=-\varepsilon x$, so
$C=\mathbb E[\varepsilon^2x^2]=\mathbb E[\tau^2x^2\cdot x^2]=3\tau^2$, using
$\mathbb E x^4=3$. Hence:

- $\mathrm{Var}\,\hat\theta_n\approx H^{-1}CH^{-1}/n=3\tau^2/n$;
- excess risk $\approx\mathrm{Tr}(H^{-1}C)/2n=3\tau^2/2n$;
- optimism $\approx3\tau^2/n$.

**4.A.3.2.** The homoscedastic formulas use $C=\tau^2H$, i.e.
$\mathrm{Tr}(H^{-1}C)=\tau^2p$ with $p=1$. They give variance $\tau^2/n$,
excess risk $\tau^2/2n$ and optimism $\tau^2/n$: all too small by a factor of
3. The noise is largest exactly where $x$ is large, which is where the data
have the most leverage on $\hat\theta$.

Numerical check (`checks/wb2b_check.py`): for a model misspecified in both
mean and noise ($y=\sin2x+(0.3+0.5|x|)\xi$ fitted linearly with an intercept),
$\mathrm{Tr}(H^{-1}C)=3.19$ while the naive $\sigma^2p=2.01$. At $n=200$,
over $2\times10^4$ replications, the excess risk is $0.00812$ against the
formula's $0.00799$, and the optimism $0.01608$ against $0.01597$, a ratio of
$1.98$. The same check for a misspecified logistic regression (a non-quadratic
loss, $n=400$) gives a ratio of $1.98$.
"""
    ),
    markdown(
        r"""
#### 4.A.4. When $C=H$: the Fisher identity and AIC

**Why.** For a probabilistic model trained with the log loss on data it
actually generated, gradient noise equals curvature. Then
$\mathrm{Tr}(H^{-1}C)=p$: every parameter costs the same, whatever the
curvature, and the formulas reduce to the classical AIC.

1. Let $\ell(\theta;x,y)=-\log p_\theta(y\mid x)$, with the data generated by $p_{\theta^*}$ (a well-specified model). Prove that $C=H$ at $\theta^*$ by differentiating $\int p_\theta(y\mid x)\,dy=1$ twice.
2. Conclude that the optimism is $p/n$ and the excess risk $p/2n$. Relate this to AIC $=-2\log L(\hat\theta)+2p$.
3. For squared loss $\ell=\tfrac12(y-f_\theta(x))^2$ with $y=f_{\theta^*}(x)+\varepsilon$, $\varepsilon$ independent of $x$ and of variance $\sigma^2$, show that $C=\sigma^2H$, so $\mathrm{Tr}(H^{-1}C)=\sigma^2p$. Check this against the exact OLS result of 1.B.2 (excess risk $\sigma^2p/(n-p-1)$ in the $(y-f)^2$ convention), minding the factor of two.
"""
    ),
    answer(
        r"""
**1.** Write $s=\nabla_\theta\log p_\theta$ (the score). Differentiating
$\int p_\theta\,dy=1$ once gives
$0=\int\nabla p_\theta=\int p_\theta\,s$, so $\mathbb E_{p_\theta}s=0$.
Differentiating again gives
$0=\int\nabla^2p_\theta=\int p_\theta\big(\nabla^2\log p_\theta+ss^\top\big)$,
using $\nabla^2p/p=\nabla^2\log p+ss^\top$. So
$\mathbb E_{p_\theta}[ss^\top]=-\mathbb E_{p_\theta}[\nabla^2\log p_\theta]$.
At $\theta=\theta^*$ the data come from $p_{\theta^*}$, so averaging also over
$x$ gives $C=\mathbb E[ss^\top]=\mathbb E[\nabla^2\ell]=H$.

**2.** $\mathrm{Tr}(H^{-1}C)=\mathrm{Tr}(I_p)=p$. So the optimism is $p/n$ and
the excess risk, which here is the KL to the truth, is $p/2n$ nats per
sample. Estimating the expected test log loss by $\hat R_n(\hat\theta_n)+p/n$
and multiplying by $2n$ gives $-2\log L(\hat\theta)+2p$: AIC is TIC with
$C=H$ assumed. This is "parameter counting", and it is exact only for
well-specified regular models.

**3.** $\nabla\ell=-r\nabla f$ with $r=y-f_\theta(x)$, and
$\nabla^2\ell=\nabla f\nabla f^\top-r\nabla^2f$. At $\theta^*$, $r=\varepsilon$.
So $C=\mathbb E[\varepsilon^2\nabla f\nabla f^\top]=\sigma^2\mathbb E[\nabla f\nabla f^\top]$
by independence, and $H=\mathbb E[\nabla f\nabla f^\top]-\mathbb E[\varepsilon\nabla^2f]=\mathbb E[\nabla f\nabla f^\top]$
since $\mathbb E[\varepsilon\mid x]=0$. Hence $C=\sigma^2H$.

For OLS with $x\sim\mathcal N(0,I)$, this predicts excess risk $\sigma^2p/2n$
in the $\tfrac12$ convention, i.e. $\sigma^2p/n$ in 1.B's $(y-f)^2$
convention. That matches the large-$n$ limit of 1.B.2's exact
$\sigma^2p/(n-p-1)$.
"""
    ),
    markdown(
        r"""
#### 4.A.5. The Bayesian side: Laplace and BIC

**Why.** A Bayesian compares models by their *evidence* (marginal
likelihood): the probability of the data averaged over the prior. To leading
order it is the maximized likelihood penalized by $\frac p2\log n$, the BIC.
The penalty counts how fast the posterior volume shrinks, which is also where
singular learning theory departs from the regular case.

Take the log loss $\ell=-\log p_\theta(y\mid x)$, so that the likelihood of
the training set is $L(\theta)=e^{-n\hat R_n(\theta)}$; a prior density $\pi$
that is continuous and positive at $\theta^*$; and the marginal likelihood
(evidence)
$$
Z_n=\int\pi(\theta)\,e^{-n\hat R_n(\theta)}\,d\theta .
$$
The *Laplace approximation* evaluates such an integral for large $n$ by
Taylor-expanding $\hat R_n$ to second order around its minimizer
$\hat\theta_n$, with Hessian $\hat H=\nabla^2\hat R_n(\hat\theta_n)$, and
integrating the resulting Gaussian exactly.

1. Use the Laplace approximation around $\hat\theta_n$ to show that
$$
-\log Z_n=n\hat R_n(\hat\theta_n)+\frac p2\log n+\tfrac12\log\det\hat H-\frac p2\log2\pi-\log\pi(\hat\theta_n)+o(1).
$$
Hence $-2\log Z_n=-2\log L(\hat\theta_n)+p\log n+O(1)$: BIC.
2. AIC's penalty is $2p$ and BIC's is $p\log n$. They answer different questions. Which question does each answer?
"""
    ),
    answer(
        r"""
**1.** Near $\hat\theta_n$,
$n\hat R_n(\theta)\approx n\hat R_n(\hat\theta_n)+\frac n2(\theta-\hat\theta_n)^\top\hat H(\theta-\hat\theta_n)$,
and $\pi(\theta)\approx\pi(\hat\theta_n)$ over the region of width
$O(n^{-1/2})$ that dominates the integral. The Gaussian integral gives
$$
Z_n\approx\pi(\hat\theta_n)\,e^{-n\hat R_n(\hat\theta_n)}\,(2\pi)^{p/2}\det(n\hat H)^{-1/2},
$$
and $\det(n\hat H)=n^p\det\hat H$. Taking $-\log$ gives the stated
expansion. Every term except the first two is $O(1)$, so
$-2\log Z_n=-2\log L(\hat\theta_n)+p\log n+O(1)$.

**2.** AIC (and TIC) estimate the expected test loss of the plug-in
predictor $p_{\hat\theta_n}$: a prediction question. BIC approximates
$-2\log$ of the evidence, so differences of BIC approximate log posterior
odds between models: a model-identification question. The evidence penalizes
each parameter by $\tfrac12\log n$ because the posterior volume shrinks like
$n^{-p/2}$ relative to the prior (the Occam factor; WB4 §2). In singular
models this volume scaling changes: $\frac p2\log n$ becomes $\lambda\log n$
with $\lambda$ the *real log canonical threshold* (RLCT), the central quantity
of singular learning theory (SLT). (This $\lambda$ is not a ridge penalty.) In
one sentence: with $K(\theta)=R(\theta)-R(\theta^*)$, the prior mass of
near-optimal parameters scales as
$\int_{K(\theta)<\epsilon}\pi(\theta)\,d\theta\propto\epsilon^{\lambda}$ as
$\epsilon\to0$ (up to a power of $\log(1/\epsilon)$). In a regular model that
set is an ellipsoid $\tfrac12\delta^\top H\delta<\epsilon$ in $p$ dimensions,
with mass $\propto\epsilon^{p/2}$, so $\lambda=p/2$. Degenerate (flat)
directions make the set fatter, which makes $\lambda$ smaller than $p/2$.
"""
    ),
    markdown(
        r"""
#### 4.A.6. Three estimands: a single fit, the Bayes ensemble, a posterior draw (optional)

**Why.** "The generalization error" of a Bayesian learner can mean three
different things, and claims from Bayesian statistics and singular learning
theory are about the second, while a trained network is closer to the first.
In the regular case they differ only by constant factors; this exercise
derives them.

Use the log loss, a model $p_\theta(y\mid x)$, and a truth $q(y\mid x)$; take
the model to be well specified, $q=p_{\theta^*}$. Let
$\pi_n(\theta)\propto\pi(\theta)\prod_ip_\theta(y_i\mid x_i)$ be the
posterior, and $\bar p_n(y\mid x)=\int p_\theta(y\mid x)\,\pi_n(d\theta)$ the
posterior predictive. With $\mathbb E_D$ over the training sample:

1. **Single-network excess risk:** $\ \mathcal E=\mathbb E_D\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,p_{\hat\theta_n}\big)=\mathbb E_D R(\hat\theta_n)-R(\theta^*)$. One deterministic output of training.
2. **Bayes generalization error:** $\ B_g=\mathbb E_D\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,\bar p_n\big)$. The posterior-predictive ensemble.
3. **Gibbs generalization error:** $\ G_g=\mathbb E_D\,\mathbb E_{\theta\sim\pi_n}\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,p_\theta\big)$. One random draw from the posterior.

Each is an expected population loss minus the entropy $R(\theta^*)$ of the
truth, so all three are in nats per sample. "Gibbs" refers to the posterior
$\propto\pi\,e^{-n\hat R_n}$ being a Gibbs (Boltzmann) distribution, of which
estimand 3 takes one sample. (Watanabe, *Mathematical Theory
of Bayesian Statistics*, 2018, ch. 1–2, defines them without the outer
$\mathbb E_D$, states results for their expectations, and treats the singular
case. For the singular Bayes generalization error and WAIC, the "widely
applicable" information criterion that estimates it in singular models, see also
[Watanabe (2010)](https://jmlr.org/papers/v11/watanabe10a.html).)

**Watanabe's regular-case results.** For a regular, well-specified model with
the ordinary posterior (temperature $T=1$), $\mathbb E\,B_g=\frac p{2n}+o(\frac1n)$ and
$\mathbb E\,G_g=\frac pn+o(\frac1n)$. Under the assumptions of singular
learning theory (the truth is in the model, and the model is analytic in
$\theta$), $\mathbb E\,B_g=\frac\lambda n+o(\frac1n)$: the real log canonical
threshold $\lambda\le p/2$ of 4.A.5 (the exponent of the prior mass of
near-optimal parameters) replaces $p/2$.

**Exercise.**

1. Show that $B_g\le G_g$ always.
2. In the regular case, approximate the posterior by $\mathcal N\big(\hat\theta_n,(nH)^{-1}\big)$. Show that $G_g\approx p/n$.
3. Show that $B_g\approx p/2n$, the same as $\mathcal E$ to leading order.
4. Without well-specification, define $K^*=\mathbb E_x\mathrm{KL}(q\|p_{\theta^*})$. Keep the same Laplace posterior and compute $G_g-K^*$ in terms of $\mathrm{Tr}(H^{-1}C)$. What does this say about the posterior's width?

<details><summary>Hint</summary>

For 1, $\mathrm{KL}(q\|\cdot)$ is convex in its second argument. For 2, write a posterior draw as $\theta=\hat\theta_n+u$, with $\mathbb E[u\mid D]=0$ and $\delta=\hat\theta_n-\theta^*$ fixed given the data. For 3, expand $\log\mathbb E_u\,p_{\hat\theta_n+u}(y\mid x)$ to second order in $u$, and use the Fisher identity of 4.A.4.
</details>
"""
    ),
    answer(
        r"""
**1.** $\bar p_n=\mathbb E_{\theta\sim\pi_n}p_\theta$, and $\mathrm{KL}(q\|\cdot)$
is convex, so by Jensen
$\mathrm{KL}(q\|\bar p_n)\le\mathbb E_{\theta\sim\pi_n}\mathrm{KL}(q\|p_\theta)$.
Average over $x$ and $D$.

**2.** For a draw $\theta=\theta^*+\delta+u$, the excess risk is
$\approx\tfrac12(\delta+u)^\top H(\delta+u)$. Given the data, $u$ has mean
zero, so the cross term vanishes in expectation, and
$$
G_g\approx\tfrac12\mathrm{Tr}\Big(H\big(\mathbb E\delta\delta^\top+\mathbb Euu^\top\big)\Big)
=\tfrac12\mathrm{Tr}\big(H(H^{-1}/n+H^{-1}/n)\big)=\frac pn,
$$
using $\mathbb E\delta\delta^\top=H^{-1}CH^{-1}/n=H^{-1}/n$ (since $C=H$).
The draw pays the ERM's error once and the posterior's own spread once more.

**3.** Write $\ell=-\log p_\theta$, $g=\nabla\ell(\hat\theta_n)$ and
$h=\nabla^2\ell(\hat\theta_n)$ at a given $(x,y)$. Then
$$
\log\bar p_n=\log\mathbb E_u\,e^{-\ell(\hat\theta_n+u)}
\approx-\ell(\hat\theta_n)+\log\mathbb E_u\,e^{-u^\top g-\frac12u^\top hu}
\approx-\ell(\hat\theta_n)+\tfrac12\mathbb E_u\big[u^\top(gg^\top-h)u\big],
$$
to order $1/n$, since $\mathbb E u=0$. With $\mathbb Euu^\top=(nH)^{-1}$, the
correction is $\frac1{2n}\mathrm{Tr}\big(H^{-1}(gg^\top-h)\big)$. Averaging over
$(x,y)\sim q$ near $\theta^*$, $\mathbb E[gg^\top-h]=C-H=0$ by the Fisher
identity. So $-\mathbb E\log\bar p_n=R(\hat\theta_n)+o(1/n)$, and
$B_g\approx\mathcal E\approx p/2n$.

In the regular case the ensemble buys nothing at leading order. The three
estimands differ by constant factors ($\tfrac12,\tfrac12,1$ times $p/n$). In
singular models they come apart: $B_g\approx\lambda/n$, while these regular-case arguments do not determine the MLE's leading error.

Numerical check (`checks/wb2b_check.py`), with linear-Gaussian regression,
$p=5$, $n=100$, known noise and a flat prior, so that the posterior is exactly
Gaussian: $\mathcal E=0.0268$ (exact finite-$n$ value
$p/2(n-p-1)=0.0266$), $B_g=0.0258$, $G_g=0.0526$, against $p/2n=0.025$ and
$p/n=0.05$.

**4.** Now $\mathbb E\delta\delta^\top=H^{-1}CH^{-1}/n$ while the posterior
still has covariance $H^{-1}/n$, so
$G_g-K^*\approx\frac1{2n}\big(\mathrm{Tr}(H^{-1}C)+p\big)$.
The approximation error $K^*$ need not vanish under misspecification. The posterior's width
is set by $H$ alone, but the actual sampling spread of $\hat\theta_n$ is set by
the sandwich. A misspecified posterior is overconfident when $C\succ H$ and
underconfident when $C\prec H$. This is one honest motivation for tempering.
"""
    ),
    markdown(
        r"""
#### 4.A.7. Which estimand do practical tools measure?

**Why.** A claim like "flat minima generalize" or "weight decay is Bayesian"
is only checkable once you know which of these quantities it is about.

The three estimands of 4.A.6 are: (1) the excess risk of a single fitted
network, (2) the error of the posterior-predictive ensemble, and (3) the error
of a single posterior draw.

For each of the following, say which of the three estimands it measures or
controls, and in which regime the link is a theorem: held-out loss, early
stopping, ridge (weight decay), and flatness $\tfrac12\mathrm{Tr}(H\Sigma)$.
"""
    ),
    answer(
        r"""
- **Held-out loss** estimates the population loss of whatever predictor you
  evaluate: one trained network (estimand 1, up to the unknown constant
  $R(\theta^*)$), an ensemble (closer to 2), or a posterior sample (3). It is
  model-free and needs no regime assumption, which is why it is the
  practical standard.
- **Early stopping** and **ridge** change the estimator; they do not choose
  an estimand. They act on the single-network excess risk by trading bias
  for variance (1.C–1.D). Ridge is MAP (maximum a posteriori: the posterior mode) under a Gaussian prior, which is neither
  the Bayes ensemble nor a Gibbs draw. (In linear-Gaussian models the
  posterior mean equals the MAP, so there it coincides with the ensemble's
  mean prediction.)
- **Flatness** $\tfrac12\mathrm{Tr}(H\Sigma)$ approximates the mean local
  excess risk when $\Sigma=\mathbb E[(\theta-\theta^*)(\theta-\theta^*)^\top]$.
  This is a second moment, equal to the covariance only for a centered parameter. If $\Sigma$ is the sampling covariance of
  $\hat\theta_n$, it is estimand 1 (4.A.2). If $\Sigma$ is the posterior's
  covariance, it is the posterior's contribution to estimand 3. If $\Sigma$ is
  SGD's stationary covariance, it is a Gibbs-like error for the "SGD
  posterior" (4.B.3). Which one is meant has to be stated.
- None of the regular-case formulas is a theorem for overparametrized
  networks trained by SGD: $H$ can be singular, and the empirical gradient
  covariance can vanish at squared-loss interpolation even when the
  population covariance does not (4.B.1).
"""
    ),
    # ------------------------------------------------------------------ §4.B
    markdown(
        r"""
<a id="wb2-section-4b"></a>

### 4.B. Does curvature cancel? Gradient noise, curvature, and SGD

**The question.** 4.A's clean results lean on $C\propto H$: gradient noise
proportional to curvature. Then a sharp direction is pinned down more tightly
by the data in exactly the proportion that errors there cost more, so
curvature cancels and only the parameter count matters. Does this hold for
trained networks? And SGD adds noise of its own: around a minimum it does not
converge but keeps jiggling, with a spread set by the learning rate and batch
size. Is that spread like a Bayesian posterior, as is sometimes claimed?

**What you get.**
- 4.B.1: per example, $C$ and $H$ are built from the same rank-one pieces,
  weighted differently. Where the network fits its training data (near
  interpolation), the gradient noise vanishes faster than the curvature.
- 4.B.2: measured diagonals of real networks follow a power law $C_{ii}\propto H_{ii}^\gamma$
  with $\gamma$ roughly 1–1.5 (in 4.B, $\gamma$ is this exponent, unrelated to
  §3's $p/n$; $\gamma=1$ is $C\propto H$, and $\gamma=0$ is noise that ignores curvature); a moment argument makes $[1,2]$ natural, though
  it does not force a fitted exponent into it.
- 4.B.3–4.B.5: on a quadratic basin, SGD's stationary covariance solves a
  linear matrix equation (a Lyapunov equation). It has the shape of a (tempered) posterior only if the
  gradient noise is isotropic ($\gamma=0$), outside the measured range; for
  $\gamma\ge1$ SGD spreads *at least as much* along sharp directions as flat
  ones, the opposite of a posterior.

**Setup, restated.** Loss $\ell(\theta;z)$, population risk $R$,
minimizer $\theta^*$ (shifted to $0$ where convenient), curvature
$H=\nabla^2R(\theta^*)$, and gradient-noise covariance
$C=\mathrm{Cov}_z[\nabla\ell(\theta;z)]$ (at $\theta^*$ unless stated). A
parameter $\theta^*+\delta$ with $\mathbb E\,\delta\delta^\top=\Sigma$ has
excess risk $\approx\tfrac12\mathrm{Tr}(H\Sigma)$ (1.D.7). For ERM in the
regular regime $\Sigma\approx H^{-1}CH^{-1}/n$ (4.A.1). For a well-specified log loss,
$C=H$ at $\theta^*$, and for squared loss with independent noise, $C=\sigma^2H$.
A *tempered posterior* at temperature $T$ is
$\pi_T(\theta)\propto\pi(\theta)\,e^{-n\hat R_n(\theta)/T}$ ($T=1$ is the
ordinary Bayesian posterior for the log loss); near a nondegenerate mode it is
approximately Gaussian with covariance $\frac TnH^{-1}$, long in flat
directions and narrow in sharp ones.

#### 4.B.1. Same pieces, different weights

**Why.** To see when $C\propto H$ can hold, write both matrices as averages
over training examples and compare the pieces. In this exercise both are
*empirical*: averages over the $n$ training examples at the current
parameter, which is what one can measure on a trained network.

*Source:* Zhang, Yang and Tu (reading in 4.B.6), §2.1, on why the Fisher
approximation $C\approx H$ fails.

1. *Squared loss.* Let $\ell_i=\tfrac12r_i^2$ with residual $r_i=f_\theta(x_i)-y_i$, and $v_i=\nabla f_\theta(x_i)$. Write the per-sample gradient $g_i$ and the per-sample Gauss–Newton matrix $h_i$ (the Hessian without the $r_i\nabla^2f$ term). Show that $h_i$ has rank at most one, and that over the training set
$$
H_{GN}=\frac1n\sum_ih_i,\qquad C=\frac1n\sum_ir_i^2h_i-\bar g\bar g^\top ,
$$
where $\bar g=\frac1n\sum_ig_i$ is the full-batch gradient and $C$ is the
empirical covariance of the $g_i$.
2. At an interpolating minimum ($r_i=0$ for all $i$), what are $C$ and $H_{GN}$? Away from interpolation, under what condition is $C=\sigma^2H_{GN}$?
3. *Logistic loss.* With $y_i\in\{0,1\}$, $s_i=\sigma(f_\theta(x_i))$, where $\sigma(t)=1/(1+e^{-t})$ is the logistic sigmoid, and $\ell_i=-y_i\log s_i-(1-y_i)\log(1-s_i)$, show that $g_i=(s_i-y_i)v_i$ and that the per-sample Gauss–Newton matrix is $h_i=a_iv_iv_i^\top$ with $a_i=s_i(1-s_i)$. For a correctly classified example with large margin, compare the weights of $v_iv_i^\top$ in $g_ig_i^\top$ and in $h_i$. Which goes to zero faster as the margin grows?
"""
    ),
    answer(
        r"""
**1.** $g_i=\nabla\ell_i=r_iv_i$ and $\nabla^2\ell_i=v_iv_i^\top+r_i\nabla^2f$,
so $h_i=v_iv_i^\top$, which has rank at most one (rank one when $v_i\ne0$). Averaging gives $H_{GN}$. The
empirical gradient covariance is
$\frac1n\sum_ig_ig_i^\top-\bar g\bar g^\top=\frac1n\sum_ir_i^2v_iv_i^\top-\bar g\bar g^\top$.

**2.** At interpolation every $r_i=0$, so every $g_i=0$ and $C=0$. But
$H_{GN}=\frac1n\sum v_iv_i^\top\neq0$. So $C\propto H$ fails completely:
gradient noise vanishes in every direction while curvature does not.
Away from interpolation, at the population optimum with $r=-\varepsilon$
independent of $x$ (hence of $v$) and $\mathbb E\varepsilon=0$:
$\bar g=0$ and $C=\mathbb E[\varepsilon^2]\,\mathbb E[vv^\top]=\sigma^2H_{GN}$.
The cancellation needs noise that is independent of where you are in input
space, i.e. a well-specified model with homoscedastic noise.

**3.** $\partial\ell_i/\partial f=s_i-y_i$, so $g_i=(s_i-y_i)v_i$, and
$\partial^2\ell_i/\partial f^2=s_i(1-s_i)$, giving $h_i=a_iv_iv_i^\top$. For
$y_i=1$ with margin $f\to\infty$: $g_ig_i^\top$ has weight
$(1-s_i)^2\approx e^{-2f}$, and $h_i$ has weight $a_i\approx e^{-f}$. So
$g_ig_i^\top\approx a_i^2v_iv_i^\top$: the noise weight is the *square* of
the curvature weight. Both vanish as the network separates the data, but the
noise vanishes faster, so $C/H\to0$ again. (If $y$ were resampled from the
model, $\mathbb E_y(s-y)^2=s(1-s)=a$ and $C=H$: the Fisher identity. The
mismatch comes from using the observed labels, which the model fits.)
"""
    ),
    markdown(
        r"""
#### 4.B.2. The diagonal power law: why between 1 and 2?

**Why.** 4.B.1 shows $C\propto H$ fails; the next question is what replaces
it. The exponent in $C_{ii}\propto H_{ii}^\gamma$ is the input 4.B.3 needs to
predict the shape of SGD's noise cloud.

Zhang, Yang and Tu (2026; reading in 4.B.6) argue that, beyond the Fisher-equals-Hessian
setting, $C\propto\mathbb E_i[h_i^2]$, where $h_i$ is the Hessian of the loss on
example $i$ and $\mathbb E_i$ averages over training examples. They also report
that $C$ and $H=\mathbb E_i[h_i]$ approximately commute ($CH\approx HC$, i.e.
they share eigenvectors) in their setting. Commutation is an additional
property, not a general consequence of the second-moment relation. In the
fully connected layers they analyse, they fit per-layer power laws
$C_{ii}\propto H_{ii}^\gamma$, with fitted $\gamma$ roughly $1.1$–$1.5$ for
cross-entropy and close to $1$ for squared loss. Their §5 derives a pair of
moment bounds and reads them as confining $\gamma$ to $[1,2]$. This exercise
derives the same kind of bounds in a simplified model and asks what they do
and do not imply.


Take rank-one per-sample Hessians $h_i=v_iv_i^\top$ as in 4.B.1.1, and the
model $C:=\mathbb E_i[h_i^2]$ (the constant of proportionality does not affect
$\gamma$).

1. Show that $C_{jj}=\mathbb E_i\big[\|v_i\|^2v_{ij}^2\big]$ and $H_{jj}=\mathbb E_i[v_{ij}^2]$.
2. Prove the envelope $H_{jj}^2\le C_{jj}\le G^2H_{jj}$, where $G^2=\max_i\|v_i\|^2$, and note that $H_{jj}\le G^2$. Draw it on log–log axes.
3. Suppose each example's gradient lies along a single coordinate: $v_i=s_{k}e_{k}$ for examples of "type" $k$, which occur with frequency $\pi_k$. Compute $H_{kk}$ and $C_{kk}$. Show that $\gamma=1$ if curvature varies across coordinates only through the frequencies $\pi_k$, and $\gamma=2$ if only through the magnitudes $s_k$.
4. Does the envelope force a fitted $\gamma$ into $[1,2]$?

<details><summary>Hint</summary>

For the lower bound in 2, use $\|v_i\|^2\ge v_{ij}^2$, then Jensen (or Cauchy–Schwarz) on $\mathbb E\,v_{ij}^4$. For 3, each example contributes to exactly one coordinate.
</details>
"""
    ),
    answer(
        r"""
**1.** $h_i^2=v_iv_i^\top v_iv_i^\top=\|v_i\|^2v_iv_i^\top$, whose $(j,j)$
entry is $\|v_i\|^2v_{ij}^2$. Average over $i$. Similarly $(h_i)_{jj}=v_{ij}^2$.

**2.** Upper bound: $\|v_i\|^2\le G^2$, so
$C_{jj}\le G^2\,\mathbb E v_{ij}^2=G^2H_{jj}$. Lower bound:
$\|v_i\|^2\ge v_{ij}^2$, so $C_{jj}\ge\mathbb E v_{ij}^4\ge(\mathbb Ev_{ij}^2)^2=H_{jj}^2$
by Cauchy–Schwarz. Also $H_{jj}\le\mathbb E\|v_i\|^2\le G^2$. On log–log axes
the points lie in a wedge between a line of slope 2 (below) and a line of
slope 1 (above), and the two lines meet at $H_{jj}=G^2$.

**3.** $H_{kk}=\pi_ks_k^2$ and $C_{kk}=\pi_ks_k^4$. If $s_k\equiv s$ and only
$\pi_k$ varies, $C_{kk}=s^2H_{kk}$: $\gamma=1$. If $\pi_k\equiv\pi$ and only
$s_k$ varies, $C_{kk}=H_{kk}^2/\pi$: $\gamma=2$. Curvature that comes from
*how often* a parameter is used gives $\gamma=1$; curvature that comes from
*how strongly* it is used on the examples that use it gives $\gamma=2$.
(Note that "all $v_i$ parallel" gives $\gamma=1$, not $2$: scaling every
gradient by a per-example factor changes all coordinates proportionally.)

**4.** No. A cloud that sits on the slope-1 line at small $H$ and on the
slope-2 line near $H=G^2$ can be fitted with a slope below 1, and the reverse
gives a slope above 2. The envelope makes $[1,2]$ natural, and 3 shows both
ends are attained by simple mechanisms. The measured range is an empirical
fact about which mechanism dominates. The checks are in `checks/wb2b_check.py`.
"""
    ),
    markdown(
        r"""
#### 4.B.3. SGD's stationary law on a quadratic basin, and the posterior

**Why.** With a constant learning rate, SGD does not converge to the minimum
but settles into a stationary cloud around it. Its covariance $\Sigma$ sets
SGD's excess risk through $\tfrac12\mathrm{Tr}(H\Sigma)$, and its shape decides
whether "SGD samples a posterior" can be true.

Near the minimum (shifted to $0$), take $R(\theta)\approx\tfrac12\theta^\top H\theta$
with $H\succ0$, and SGD with learning rate $\eta$ and batch size $B$:
$$
\theta_{k+1}=\theta_k-\eta\,(H\theta_k+\xi_k),\qquad\mathbb E\,\xi_k=0,\quad\mathrm{Cov}\,\xi_k=C/B,
$$
with $\xi_k$ independent across steps and independent of $\theta_k$, and
$C$ fixed (state-independent near the minimum). Here $\xi_k$ is the
minibatch gradient's deviation from the full gradient; averaging $B$ examples
divides the per-example covariance $C$ by $B$. For discrete-time stationarity
assume $0<\eta<2/\lambda_{\max}(H)$. The *stationary covariance* $\Sigma$ is
the covariance of $\theta_k$ after the iteration has run long enough to
forget its starting point, so that it no longer changes from step to step.

Two named objects appear. A *Lyapunov equation* is a linear matrix equation
$A\Sigma+\Sigma A^\top=Q$ for an unknown symmetric $\Sigma$; it is the
equation for the stationary covariance of any linear system driven by noise.
An *Ornstein–Uhlenbeck (OU) process* is the continuous-time version of such a
system: a particle pulled linearly back toward $0$ and kicked by Gaussian
white noise, $d\theta=-A\theta\,dt+Q^{1/2}dW$, with $W$ a standard Brownian
motion. Noisy gradient descent on a quadratic basin, with small steps, is one.

*Source:* this stationary-distribution analysis, and the idea of tuning
constant-step SGD so that its stationary law approximates a posterior, are in
[Mandt, Hoffman and Blei (2017)](https://arxiv.org/abs/1704.04289).


1. Show that the stationary covariance of the recursion satisfies $\Sigma_\eta=(I-\eta H)\Sigma_\eta(I-\eta H)+\eta^2C/B$, and that to leading order in $\eta$ this is the Lyapunov equation
$$
H\Sigma+\Sigma H=\frac\eta B\,C .
$$
(It is also the stationary equation of the Ornstein–Uhlenbeck process $d\theta=-H\theta\,dt+\sqrt{\eta/B}\,C^{1/2}dW$, with $t=\eta k$.)
2. Show that the Lyapunov equation has a unique solution for $H\succ0$. If $C$ and $H$ commute, show that $\Sigma=\frac\eta{2B}CH^{-1}$. With $C=cH^\gamma$, this is $\Sigma=\frac{\eta c}{2B}H^{\gamma-1}$.
3. Show that $\Sigma=\frac TnH^{-1}$ for some temperature $T$ if and only if $C\propto I$. Do not assume that $C$ and $H$ commute. Find $T$.
4. Show that $\tfrac12\mathrm{Tr}(H\Sigma)=\frac{\eta\,\mathrm{Tr}\,C}{4B}$ exactly, for any $C$.
5. With $C=cH^\gamma$, write three excess risks: ERM's $\frac1{2n}\mathrm{Tr}(H^{-1}C)$ from 4.A.2, SGD's stationary value, and the tempered posterior's $\tfrac12\mathrm{Tr}(H\cdot\frac TnH^{-1})$. What does each count?
6. *Preconditioning.* Replace the update by $\theta\leftarrow\theta-\eta P(H\theta+\xi)$ with $P\succ0$. Write the new Lyapunov equation. Assuming $C\succ0$, show that $P=C^{-1}$ gives $\Sigma=\frac\eta{2B}H^{-1}$.
7. For $\gamma\in[1,2]$, compare the shapes of $\Sigma$ and the posterior: along which directions does SGD spread more than the posterior? What would "SGD is approximately Bayesian" have to mean?

<details><summary>Hint</summary>

For 2, $\Sigma=\int_0^\infty e^{-Ht}\,\frac\eta BC\,e^{-Ht}dt$ solves the equation; for uniqueness, work in $H$'s eigenbasis, where the equation reads $(h_j+h_k)\Sigma_{jk}=\frac\eta BC_{jk}$. For 3, substitute $\Sigma=\kappa H^{-1}$. For 4, take the trace.
</details>
"""
    ),
    answer(
        r"""
**1.** $\theta_k$ is independent of $\xi_k$, so
$\mathrm{Cov}\,\theta_{k+1}=(I-\eta H)\,\mathrm{Cov}\,\theta_k\,(I-\eta H)+\eta^2C/B$.
At stationarity, expanding gives
$\Sigma_\eta=\Sigma_\eta-\eta(H\Sigma_\eta+\Sigma_\eta H)+\eta^2H\Sigma_\eta H+\eta^2C/B$,
i.e. $H\Sigma_\eta+\Sigma_\eta H=\frac\eta BC+\eta H\Sigma_\eta H$. Since
$\Sigma_\eta=O(\eta)$, the last term is $O(\eta^2)$; dropping it gives the
Lyapunov equation. (Along an eigendirection of curvature $h$, the discrete
law is wider by a relative $O(\eta h)$, and diverges at $\eta h\ge2$.)

**2.** In the eigenbasis of $H$ (eigenvalues $h_j>0$), the equation reads
$(h_j+h_k)\Sigma_{jk}=\frac\eta BC_{jk}$, which has the unique solution
$\Sigma_{jk}=\frac\eta B\,C_{jk}/(h_j+h_k)$. If $C$ commutes with $H$, $C$ is
diagonal in the same basis, so $\Sigma_{jj}=\frac\eta{2B}C_{jj}/h_j$, i.e.
$\Sigma=\frac\eta{2B}CH^{-1}$. With $C=cH^\gamma$,
$\Sigma=\frac{\eta c}{2B}H^{\gamma-1}$.

**3.** If $\Sigma=\kappa H^{-1}$, then $H\Sigma+\Sigma H=2\kappa I$, so
$\frac\eta BC=2\kappa I$ and $C\propto I$. Conversely, if $C=cI$ then
$\Sigma=\frac{\eta c}{2B}H^{-1}$ solves the equation, and by uniqueness it is
the solution. No commutation was assumed. Matching $\frac{\eta c}{2B}=\frac Tn$
gives $T=\frac{n\eta c}{2B}$.

**4.** Take the trace of the Lyapunov equation:
$\mathrm{Tr}(H\Sigma)+\mathrm{Tr}(\Sigma H)=2\,\mathrm{Tr}(H\Sigma)=\frac\eta B\mathrm{Tr}\,C$.
So $\tfrac12\mathrm{Tr}(H\Sigma)=\frac{\eta\,\mathrm{Tr}\,C}{4B}$, for any $C$,
commuting or not.

**5.** With $C=cH^\gamma$:

| | excess risk | counts |
|---|---|---|
| ERM (4.A.2) | $\frac c{2n}\mathrm{Tr}\,H^{\gamma-1}$ | sampling noise in $\hat\theta_n$, each direction weighted by noise over curvature |
| SGD stationary | $\frac{\eta c}{4B}\mathrm{Tr}\,H^\gamma$ | total gradient noise per step; curvature does not enter separately |
| posterior | $\frac{Tp}{2n}$ | parameter count at temperature $T$; neither $H$ nor $C$ enters |

At $\gamma=1$ (the Fisher case), ERM gives $\frac{cp}{2n}$, parameter
counting, while SGD gives $\frac{\eta c}{4B}\mathrm{Tr}H$, which grows with
sharpness. For this $\gamma=1$ comparison at fixed $c$, the SGD excess-risk value increases with curvature. This local calculation does not by itself determine how SGD selects between basins. Its noise scale is
$\eta/B$, not $1/n$.

**6.** The update is $\theta\leftarrow(I-\eta PH)\theta-\eta P\xi$. The same
calculation gives $PH\Sigma+\Sigma HP=\frac\eta BPCP$. With $P=C^{-1}$,
try $\Sigma=\kappa H^{-1}$: the left side is $2\kappa C^{-1}$ and the right
side is $\frac\eta BC^{-1}$, so $\kappa=\frac\eta{2B}$. Preconditioning by the
inverse noise covariance makes the continuous-time stationary covariance
posterior-shaped when $C\succ0$. With Gaussian diffusion this also matches
the Gaussian stationary law; covariance matching alone does not establish
a Gaussian law for arbitrary discrete update noise. This is the idea behind using preconditioned SGD as an approximate
posterior sampler.

**7.** With $\gamma\in[1,2]$, $\Sigma\propto H^{\gamma-1}$ ranges from
isotropic ($\gamma=1$) to $\propto H$ ($\gamma=2$). The posterior
$\propto H^{-1}$ is long in flat directions and narrow in sharp ones. SGD's
cloud is *at least as wide in sharp directions as in flat ones*: the
opposite ordering. "SGD samples a (tempered) posterior" is the claim that
$\gamma\approx0$, i.e. isotropic gradient noise, or that one has
preconditioned by $C^{-1}$. Neither holds for plain SGD in the measured
regime.
"""
    ),
    markdown(
        r"""
#### 4.B.4. The best temperature, and a symmetry (optional)

**Why.** If SGD's cloud $\mathcal N(0,\Sigma)$ is not shaped like the
posterior, how close can the best-tempered posterior $\mathcal N(0,\frac TnH^{-1})$
(the Laplace approximation of 4.B's setup) get? The
Kullback–Leibler divergence $\mathrm{KL}$ measures the mismatch, and its minimum
over $T$ isolates the part that no choice of temperature can fix: the shape.

1. Show that $T^*=n\,\mathrm{Tr}(H\Sigma)/p$ minimizes $\mathrm{KL}\big(\mathcal N(0,\Sigma)\,\|\,\mathcal N(0,\frac TnH^{-1})\big)$ over $T$, that the minimum is
$$
\mathrm{KL}^*=\tfrac12\big[p\log\big(\mathrm{Tr}(H\Sigma)/p\big)-\log\det(H\Sigma)\big]\ \ge0,
$$
with equality iff $H\Sigma\propto I$, and that at $T=T^*$ the posterior's excess risk $\frac{T^*p}{2n}$ equals SGD's $\tfrac12\mathrm{Tr}(H\Sigma)$.
2. In the commuting case with $p=2$ and $H=\mathrm{diag}(1,\kappa)$, show that $\mathrm{KL}^*=\log\cosh\big(\tfrac\gamma2\log\kappa\big)$, which is symmetric in $\gamma\mapsto-\gamma$. Is $\mathrm{KL}^*$ symmetric in $\gamma$ for $p=3$?

<details><summary>Hint</summary>

For zero-mean Gaussians, $\mathrm{KL}(\mathcal N(0,\Sigma)\,\|\,\mathcal N(0,S))=\tfrac12\big[\mathrm{Tr}(S^{-1}\Sigma)-p+\log\det S-\log\det\Sigma\big]$. Minimize over $\log T$, then write the result in terms of the eigenvalues of $H\Sigma$ and compare their arithmetic and geometric means. For 2, in the commuting case $H\Sigma\propto H^\gamma$.
</details>
"""
    ),
    answer(
        r"""
**1.** For Gaussians with mean zero,
$\mathrm{KL}(\mathcal N(0,\Sigma)\|\mathcal N(0,S))=\tfrac12[\mathrm{Tr}(S^{-1}\Sigma)-p+\log\det S-\log\det\Sigma]$.
With $S=\frac TnH^{-1}$:
$$
\mathrm{KL}(T)=\tfrac12\Big[\tfrac nT\mathrm{Tr}(H\Sigma)-p+p\log\tfrac Tn-\log\det(H\Sigma)\Big].
$$
$\partial_T\mathrm{KL}=\tfrac12[-\tfrac n{T^2}\mathrm{Tr}(H\Sigma)+\tfrac pT]=0$
gives $T^*=n\,\mathrm{Tr}(H\Sigma)/p$; the function is convex in $\log T$, so
this is the minimum. Substituting gives the stated $\mathrm{KL}^*$. With
$\mu_j$ the eigenvalues of $H\Sigma$ (positive, since $H\Sigma$ is similar to
$H^{1/2}\Sigma H^{1/2}$), $\mathrm{KL}^*=\frac p2[\log\bar\mu_{\rm arith}-\log\bar\mu_{\rm geom}]\ge0$
by AM–GM, with equality iff all $\mu_j$ are equal, i.e. $H\Sigma\propto I$.
Finally, $\frac{T^*p}{2n}=\tfrac12\mathrm{Tr}(H\Sigma)$. The KL-optimal
temperature is also the one that matches excess risks.

**2.** Commuting case: $H\Sigma\propto H^\gamma$ has eigenvalues
$\propto(1,\kappa^\gamma)$. With $x=1$, $y=\kappa^\gamma$,
$\mathrm{KL}^*=\log\frac{(x+y)/2}{\sqrt{xy}}=\log\cosh\big(\tfrac12\log\frac yx\big)=\log\cosh\big(\tfrac\gamma2\log\kappa\big)$,
which is even in $\gamma$. For $p=3$ it is not symmetric in general: the
arithmetic mean of $\mu^\gamma$ and of $\mu^{-\gamma}$, each relative to its
geometric mean, differ unless the log-spectrum is symmetric. For $H$ with
eigenvalues $(1,2,30)$, $\mathrm{KL}^*(1.5)=2.97$ while $\mathrm{KL}^*(-1.5)=1.88$
(`checks/wb2b_check.py`). The interactive figure after 4.B.5 uses $p=2$, so the
symmetry of its $\mathrm{KL}^*$ readout is a feature of $p=2$.
"""
    ),
    markdown(
        r"""
#### 4.B.5. A two-dimensional example

**Why.** To see the geometry of 4.B.3 at a glance: which way SGD's cloud
points compared with the posterior's.

Take $H=\mathrm{diag}(1,10)$: $\theta_1$ is flat and $\theta_2$ sharp. With
gradient noise $C=cH^\gamma$, SGD's stationary covariance is
$\Sigma=\frac{\eta c}{2B}H^{\gamma-1}$ (4.B.3.2) and the posterior's is
$\frac TnH^{-1}$. Use these to find the axes and aspect ratio of SGD's
$\Sigma$ for $\gamma=0,1,2$, and compare each with the posterior's
$\propto H^{-1}$. Which way does each SGD ellipse point? What happens if $C$'s
eigenbasis is rotated relative to $H$'s? The interactive figure below draws
these ellipses.
"""
    ),
    answer(
        r"""
- $\gamma=0$: $\Sigma\propto H^{-1}$, elongated along the flat $\theta_1$ with
  aspect ratio $\sqrt{10}$ in standard deviations. Same shape as the
  posterior.
- $\gamma=1$: $\Sigma=\frac{\eta c}{2B}I$, a circle.
- $\gamma=2$: $\Sigma\propto H$, elongated along the *sharp* $\theta_2$:
  rotated $90°$ from the posterior.
- Rotating $C$'s eigenbasis tilts $\Sigma$ off the axes. Since
  $\Sigma\propto H^{-1}$ requires $C\propto I$ (4.B.3.3), no temperature
  fixes the shape, and $\mathrm{KL}^*>0$ for every $T$.
- The discrete-time ellipse is slightly wider than the continuous one along
  $\theta_2$, by a relative $O(\eta h_2)=O(10\eta)$, where $h_2=10$ is the sharp curvature.

This is as far as WB2 goes. The Training Dynamics materials take up
state-dependent noise, edge-of-stability effects and what happens away from a
quadratic basin.
"""
    ),
    markdown(
        figure(
            "assets/interactive/sgd_stationary_vs_posterior.html",
            "Slide γ from 0 to 2 and watch the SGD ellipse rotate from the posterior's shape through a circle "
            "to the opposite orientation. Then rotate the noise (φ) and check that no temperature recovers the "
            "posterior shape. The readout panel shows ½Tr(HΣ) = ηTr C/4B, T*, and the discrete-time correction.",
        )
    ),
    markdown(
        r"""
#### 4.B.6. Readings and rabbit holes

"""
        + reading(
            "On the superlinear relationship between SGD noise covariance and loss landscape curvature",
            "https://arxiv.org/abs/2602.05600",
            "Yikuan Zhang, Ning Yang, Yuhai Tu (arXiv, 2026)",
            "The assumption that SGD noise covariance is proportional to the Hessian holds only under "
            "restrictive conditions that deep networks typically violate. Using an activity–weight duality, "
            "they find C ∝ E_p[h_p²] instead. They also report that C and H approximately commute without "
            "coinciding (a separate property; see 4.B.2), and that the diagonals follow per-layer power laws C_ii ∝ H_ii^γ, with fitted γ near 1–1.5 "
            "and an argument that 1 ≤ γ ≤ 2.",
            "Read §2 (why the Fisher-equals-Hessian argument fails: compare 4.B.1) and skim §3, whose "
            "§3.4 derives C ∝ E[h_p²]. Then read §4 (the measured power laws, on MLPs and small CNNs on "
            "MNIST and CIFAR-10) and §5 (the moment bounds on γ, which are the envelope of 4.B.2, and why "
            "cross-entropy gives larger γ than squared loss). Note which layers the fits cover before "
            "carrying γ into 4.B.3.",
            importance=4, length=2,
        )
        + "\n\n"
        + reading(
            "A circuit prior in NN-Bayes",
            "https://www.lesswrong.com/posts/SqDHeuycNkERurtSc/a-circuit-prior-in-nn-bayes",
            "Kaarel Hänni, Dmitry Vaintrob (LessWrong, 2026)",
            "Bayesian learning with neural networks has a circuit prior: when the target is computed by a "
            "small circuit, few samples suffice for good test accuracy, even for arbitrarily overparametrized "
            "networks, under particular prior scalings and caveats.",
            "A post with slides presenting a simplified version of the result, and open problems at the end. "
            "It is a statement about Bayesian learning (4.A.6), not about SGD. The comment thread "
            "discusses exactly that gap (some circuit-learnable functions, like sparse parities, are hard "
            "for gradient descent). Read it with 4.B.3.7 in mind.",
            importance=1, length=2,
        )
    ),
    # ------------------------------------------------------------------ §4.C
    markdown(
        r"""
<a id="wb2-section-4c"></a>

### 4.C. Fantastic generalization measures

**The question.** Dozens of quantities have been proposed as explanations of
why one network generalizes better than another: parameter norms, margins,
flatness (the curvature $H$ of the loss at the trained parameters), PAC-Bayes
bounds (2.E), gradient noise (the per-example gradient covariance $C$). Which of them actually
predict the generalization gap when you vary training in controlled ways? This
subsection is one reading, plus an exercise connecting its findings to the
theory of 1.D.7, 4.A and 4.B: which measures are invariant under
reparametrization (changes of parameters that leave the network function
unchanged), and why the assumptions behind TIC fail for the networks in the
study. TIC, recall, estimates the generalization gap of a fitted model as
$\frac1n\mathrm{Tr}(H^{-1}C)$ (4.A.2), assuming fixed $p$, $n\to\infty$ and a
nondegenerate minimum.

"""
        + reading(
            "Fantastic generalization measures and where to find them",
            "https://arxiv.org/abs/1912.02178",
            "Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, Samy Bengio (ICLR 2020)",
            "A large controlled study: 2187 CIFAR-10 networks spanning 7 hyperparameters (over 10,000 "
            "networks with repeats and SVHN), all trained to the same training cross-entropy, 0.01, and "
            "more than 40 published complexity measures ranked by how well they predict the generalization gap. "
            "Many norm-based measures correlate negatively with the gap when the optimizer is stochastic, "
            "the product of layer spectral norms strongly so. Sharpness-based measures (PAC-Bayes bounds "
            "and Keskar et al.'s sharpness) do best overall, and measures of gradient noise and of "
            "optimization speed are also predictive.",
            "§2 defines the rank correlations, including a 'granulated' version that varies one "
            "hyperparameter at a time, and a conditional-independence test; §4 has the results; the "
            "appendix defines every measure.",
            importance=5, length=3,
        )
        + r"""

#### 4.C.1. Invariance, and the assumptions behind TIC

1. Let $\theta=A\phi$ with $A$ invertible be a reparametrization that leaves the network function fixed (or the linearization of a smooth one at the minimum). Show that at the minimum $H\to A^\top HA$ and $C\to A^\top CA$, and that $\mathrm{Tr}(H^{-1}C)$ is invariant. Which of the parameter $\ell_2$ norm, the gradient variance $\mathrm{Tr}\,C$, $\mathrm{Tr}\,H$ and the product of layer spectral norms are invariant under the layer rescaling of 1.D.7.3, and which under a general $A$?
2. Jiang et al. train every model to the same cross-entropy. Why? Using 4.B.1, what does this do to the per-sample gradients and to $\hat C$? Which assumptions behind the derivation of TIC in 4.A.2 fail for these networks?
3. Jiang et al. score each measure by rank correlation with the gap. *Kendall's $\tau$* over a set of models is the fraction of pairs of models that the measure and the gap order the same way, minus the fraction they order oppositely; it lies in $[-1,1]$, and $1$ means the measure ranks the models exactly as the gap does. The *pooled* $\tau$ uses all pairs of models. The *granulated* $\Psi$ (their §2.2) computes $\tau$ only over pairs that differ in a single hyperparameter, then averages over hyperparameters. Why might a measure have a high pooled $\tau$ but a low $\Psi$?

<details><summary>Hint</summary>

For 1, per-sample gradients transform as $\nabla_\phi\ell=A^\top\nabla_\theta\ell$, and at a critical point the Hessian transforms as in 1.D.7.4. For 2, list the assumptions of 4.A's regular regime (fixed $p$ with $n\to\infty$, $H\succ0$, the trained parameter is the ERM) and check each.
</details>
"""
    ),
    answer(
        r"""
**1.** Under a reparametrization $\theta=A\phi$ with $A$ invertible (or its
linearization, for a smooth one) that leaves the function fixed, gradients
transform as $\nabla_\phi=A^\top\nabla_\theta$, so $H\to A^\top HA$ and
$C\to A^\top CA$ at the minimum. Then
$\mathrm{Tr}(H^{-1}C)\to\mathrm{Tr}(A^{-1}H^{-1}A^{-\top}A^\top CA)=\mathrm{Tr}(H^{-1}C)$:
TIC is invariant. (More generally, the eigenvalues of $H^{-1}C$ are.)

- *Not invariant:* the parameter $\ell_2$ norm, the raw gradient variance
  $\mathrm{Tr}\,C$, and plain sharpness ($\mathrm{Tr}\,H$ or a worst-case
  perturbation of fixed size).
- *Invariant to the layer rescaling of 1.D.7.3 but not to general
  reparametrizations:* the product of layer spectral norms, since
  $\|\alpha W_1\|\,\|W_2/\alpha\|=\|W_1\|\,\|W_2\|$ in a ReLU network.
- *Functions of the function:* the margin, and the gap itself.

**2.** Fixing the training loss removes a trivial predictor: models with
higher training loss have a different gap for uninteresting reasons. The
price is that every model is near interpolation, where per-sample gradients
are small (4.B.1). $\hat C$ is then small and dominated by the hardest
examples, and the regular-case theory behind $\mathrm{Tr}(H^{-1}C)$ is
furthest from applying. The assumptions that fail:
- *Interpolation* (4.B.1): at exact squared-loss interpolation, the empirical
  $\hat C$ is zero even though population gradient variability and test
  loss can remain positive. Plug-in TIC then loses its regular-regime
  justification. Jiang et al. stop at a small positive cross-entropy,
  not exact interpolation; for cross-entropy both curvature and gradient
  covariance can shrink, at different rates.
- *Singular $H$*: overparametrized networks have flat directions, so
  $H^{-1}$ is undefined and damped versions depend on $\epsilon$. The
  regular-case $p/2$ becomes the RLCT (4.A.5–4.A.6), which TIC does not see.
- *SGD is not ERM*: the trained parameter's spread is the stationary law of
  4.B.3, set by $\eta/B$ and $\mathrm{Tr}\,C$, not the sandwich. Learning-rate
  and batch-size axes move the gap through a channel TIC does not model.
  ($\mathrm{Tr}\,C$ itself does model it, which is one reading of Jiang et
  al.'s gradient-variance result.)
- *Wrong loss*: TIC is about the training loss. The 0–1 gap can move while
  the cross-entropy optimism does not, and conversely.
- *Ensembles*: TIC is about a single fitted network. If what you report is
  an ensemble's error, the leading constant can change (4.A.6).

Where the assumptions do roughly hold ($p$ small compared with $n$, a
nondegenerate minimum, training loss bounded away from zero, a trained
parameter close to the ERM), comparisons within one architecture across data
size or regularization strength are the natural use of TIC.

**3.** If one hyperparameter (say, width) moves both the measure and the gap
strongly, the pooled $\tau$ is high even if the measure tracks nothing within
any fixed setting of the others. $\Psi$ only counts comparisons where one
hyperparameter changes, so a measure that just tracks one axis scores well on
that axis and near zero on the others. A measure that predicts along one axis
and not another is tracking the axis, not the gap.
"""
    ),
]

