"""WB2 §1.A–1.D: opening, risks, least squares, bias–variance, regularization.

Old formulas are checked numerically by checks/wb2a_check.py.
"""
from nbkit import *  # noqa: F401,F403
from wb2_new_regularization import OPENER as REG_OPENER, L1 as REG_L1, NOISE as REG_NOISE  # noqa: F401

OPENING = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# Workbook 2 — Statistical learning and regularization: why the hacky bullshit works

**Mysteries of Deep Learning, Iliad Intensive 2026**

Statistical learning theory is the mathematics of learning a rule from finitely
many examples and asking how well the rule does on examples it has not seen.
Deep learning practice is full of tricks that look like hacks from the outside:
weight decay, early stopping, dropout, data augmentation, "flat minima",
training straight through the point where the training loss hits zero. This
workbook builds the classical theory in models simple enough to solve exactly,
mostly linear regression, and uses it to say what each trick does and what it
does not. The aim is that when you meet a claim about generalization, you can
say which classical statement it borrows, and in which regime that statement
is a theorem.

**How to use it.** There are four sections. Pick whatever interests you, in any
order; nobody is expected to finish. Each section and subsection opens with a
short pitch saying what question it answers, and restates the setup it needs,
so you can start almost anywhere. §1.A–1.C are the common background; if the
words "excess risk", "bias–variance" and "ridge" are already familiar, skim
them. Derivations are exercises: a group such as **1.D.3** has numbered parts,
and "1.D.3.2" means part 2 of it. Where a result is quoted rather than derived,
the text says so, and each subsection links the textbook sections or papers its
derivations come from.

**Worked solutions.** Every problem has a worked solution folded directly
after it under "Solution"; click to reveal it when you want it. Solutions are
for everyone. Hints are folded under "Hint".

| Section | What is in it |
|---|---|
| [1. Classical statistical learning](#wb2-section-1) | **1.A** kinds of learning problem (supervised, unsupervised, self-supervised, ...); risk, empirical risk minimization, and two different things called "generalization". **1.B** least squares, solved exactly. **1.C** bias–variance. **1.D** regularization: a survey of the techniques, then, in linear regression, ridge as a prior and as a constraint; $L^1$ and sparsity; ridge as a spectral filter, and early stopping; input noise and data augmentation; dropout; when flatness can matter, and the "sharp minima can generalize" reading. **1.E** kernels and the kernel trick: kernel ridge regression, random features, Gaussian processes. **1.F** information theory and rate–distortion; the information bottleneck; the debate over whether deep networks compress. |
| [2. Doesn't some theorem make this impossible?](#wb2-section-2) | No free lunch; universal approximation; VC dimension and capacity bounds; Zhang et al.'s random-label experiments, which make class-wide bounds vacuous; PAC-Bayes, which bounds the function you actually found. |
| [3. Overfitting and double descent](#wb2-section-3) | Minimum-norm least squares across the interpolation threshold; why optimal ridge removes the peak; benign overfitting; which regime LLMs are in. |
| [4. Bayes, MLE, and generalization measures](#wb2-section-4) | One story about how much a fitted model loses on new data, to leading order: the sandwich covariance, AIC, TIC and BIC; a single fit versus the Bayes ensemble; whether gradient noise and curvature cancel, and what SGD's noise does; which published measures actually predict generalization. |

**Notation used throughout.** Each subsection restates what it uses; this is the
common core.

- A data point is $z=(x,y)$: an input $x$ and a target or label $y$. Data are drawn i.i.d. from an unknown distribution $P$. The training sample is $S=(z_1,\dots,z_n)$, so $n$ is the number of training examples.
- A model is a family of predictors $f_\theta$ with parameters $\theta\in\mathbb R^p$, so $p$ is the number of parameters. A loss $\ell(\theta;z)$ scores one prediction: for example the squared loss $(y-f_\theta(x))^2$, or the log loss $-\log p_\theta(y\mid x)$ for a model that outputs probabilities.
- The **population risk** (test loss) is $R(\theta)=\mathbb E_{z\sim P}\,\ell(\theta;z)$. The **empirical risk** (training loss) is $\hat R_n(\theta)=\frac1n\sum_{i=1}^n\ell(\theta;z_i)$.
- **Empirical risk minimization** (ERM) returns $\hat\theta_n\in\arg\min_\theta\hat R_n(\theta)$. The **population minimizer** $\theta^*\in\arg\min_\theta R(\theta)$ is the best the model can do with infinite data; it need not be the truth.
- The running example is the linear model $y=x^\top\beta+\varepsilon$, with noise $\varepsilon$ of mean $0$ and variance $\sigma^2$, independent of $x$. Stacking the inputs as rows gives the design matrix $X\in\mathbb R^{n\times p}$, and the targets form $y\in\mathbb R^n$. The ratio $\gamma=p/n$ matters in §3.
- *Ridge regression* with penalty $\lambda>0$ is $\hat\beta_\lambda=\arg\min_b\frac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y$. The thin singular value decomposition $X=\sum_k s_k u_kv_k^\top$ has singular values $s_k>0$ and orthonormal singular vectors $u_k\in\mathbb R^n$, $v_k\in\mathbb R^p$; directions $v_k$ with small $s_k$ are the ones the data barely determine.
- Logarithms are natural, so information is in nats.
"""
    ),
]

S1ABC = [
    # ------------------------------------------------------------------ §1
    markdown(
        r"""
<a id="wb2-section-1"></a>

## 1. Classical statistical learning

This is the textbook theory of learning from examples, from before deep
learning: what "generalization" means, how to compute it exactly in linear
regression, how the error splits into bias and variance, what regularization
does, how kernels make linear methods nonlinear, and one information-theoretic
account of what a good representation is. Most of it is solvable in closed
form, which is the point: it gives exact reference cases against which claims
about deep networks can be checked. Everything later in the workbook uses the
vocabulary of 1.A–1.C.

<a id="wb2-section-1a"></a>

### 1.A. Learning problems, risks, and two kinds of generalization

**The question.** "Deep networks generalize well" is said about two different
quantities, and papers often switch between them. This subsection sets up the
framework and separates them.

**Kinds of learning problem.** The traditional breakdown is by what data the
learner sees and what it is asked to do.

| Paradigm | Data | Goal | Examples |
|---|---|---|---|
| Supervised | Pairs $(x,y)$ | Predict $y$ from $x$: regression (real $y$) or classification (discrete $y$) | Linear regression, image classification |
| Unsupervised | Inputs $x$ only | Model the distribution $p(x)$, or find structure in it: density estimation, clustering, dimensionality reduction | PCA, $k$-means, mixture models, VAEs, diffusion models |
| Self-supervised | Inputs $x$ only, with targets manufactured from $x$ itself | Supervised learning on those targets, usually to learn a representation or a generative model | Next-token prediction (LLM pretraining), masked-token prediction, contrastive learning |
| Semi-supervised | A few labelled pairs, many unlabelled inputs | Use the structure of $p(x)$ to predict $y$ better | Pseudo-labelling, consistency training |
| Reinforcement | States, actions and rewards from interacting with an environment | Find a policy that maximizes expected reward | Games, robotics, RL fine-tuning of LLMs |

Two other distinctions cut across the table. A *discriminative* model learns
$p(y\mid x)$ or a predictor of $y$; a *generative* model learns $p(x)$ or
$p(x,y)$ and can sample new data. In *batch* (offline) learning the whole
dataset is fixed in advance; in *online* learning examples arrive one at a
time, and in RL the learner's own actions decide which data it sees.

Statistical learning theory is mostly a theory of the supervised, i.i.d., batch
case, and so is this workbook. That covers more than it sounds. Density
estimation with the log loss is the same framework with $\ell(\theta;x)=-\log
p_\theta(x)$ and no label. LLM pretraining is supervised prediction of each
token from its context (WB3), though the tokens of one document are not i.i.d.
RL breaks the framework more seriously, since the data distribution depends on
the policy being learned.

**Risks.** In the supervised setting, data $z=(x,y)$ are drawn i.i.d. from an
unknown $P$, and a learning rule maps the training sample
$S=(z_1,\dots,z_n)$ to parameters $\hat\theta_n$. The quality of
$\hat\theta_n$ is its *population risk* $R(\hat\theta_n)=\mathbb
E_{z\sim P}\,\ell(\hat\theta_n;z)$, the expected loss on a fresh example. The
learner can compute only the *empirical risk* $\hat R_n(\theta)=\frac1n\sum_i
\ell(\theta;z_i)$, the average loss on the training sample. The commonest rule
is empirical risk minimization (ERM): $\hat\theta_n\in\arg\min\hat R_n$. The
benchmark is the population minimizer $\theta^*\in\arg\min R$, the best
parameter in the model.

**Two quantities.** For any learning rule, not only ERM, define the
*generalization gap* and the *excess risk*:

$$
\mathrm{gap}_n=R(\hat\theta_n)-\hat R_n(\hat\theta_n),
\qquad
\mathcal E_n=R(\hat\theta_n)-R(\theta^*).
$$

The gap compares test loss with training loss: is the training loss an honest
estimate of how the model will do? The excess risk compares the fitted model
with the best one available: did the learner find a good model? Both are random
through $S$. The exercises below show that either can be large while the other
is zero; that for ERM the expected gap bounds the expected excess risk; and
that a held-out set measures one of them but not the other. 1.B computes both
exactly for least squares.

#### 1.A.1. Gap without excess, excess without gap

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

Memorizing costs nothing here because the memorized inputs never recur. This is a caricature of benign overfitting; 3.D has a real version.

**2.** Take the rule that ignores the data and outputs a fixed $\theta_0$ with $R(\theta_0)>R(\theta^*)$. Since $\theta_0$ does not depend on $S$, $\mathbb E\hat R_n(\theta_0)=R(\theta_0)$, so $\mathbb E\,\mathrm{gap}_n=0$. Meanwhile $\mathcal E_n=R(\theta_0)-R(\theta^*)>0$ for every $S$. A small gap means the training loss is an honest estimate of the test loss, not that the model is good.

**3.** Write

$$
\mathcal E_n=\underbrace{R(\hat\theta_n)-\hat R_n(\hat\theta_n)}_{\mathrm{gap}_n}
+\underbrace{\hat R_n(\hat\theta_n)-\hat R_n(\theta^*)}_{\le 0\ \text{(ERM)}}
+\underbrace{\hat R_n(\theta^*)-R(\theta^*)}_{\text{mean }0}.
$$

The middle term is non-positive because $\hat\theta_n$ minimizes $\hat R_n$. The last has mean zero because $\theta^*$ does not depend on $S$. Taking expectations gives $\mathbb E\,\mathcal E_n\le\mathbb E\,\mathrm{gap}_n$. Bounding the first and third terms by $\sup_\theta|R-\hat R_n|$ and dropping the middle one gives $\mathcal E_n\le2\sup_\theta|R(\theta)-\hat R_n(\theta)|$. This is the route every capacity bound takes (§2.C): control the worst gap over the whole class, and the excess risk follows.

**4.** Because $\hat\theta_n$ is independent of the held-out data, the held-out mean loss is unbiased for $R(\hat\theta_n)$, with standard error $\mathrm{sd}(\ell)/\sqrt m$. Subtracting the training loss estimates the gap. The excess risk also needs $R(\theta^*)$, the best achievable loss in the class, which a held-out loss alone does not identify without further assumptions or knowledge. You can measure *differences* of excess risk between two fitted predictors in the same model class, since $R(\theta^*)$ cancels. Different classes can have different optimum risks. More generally, a held-out set estimates the population loss of whatever predictor you evaluate on it, whether one network or an ensemble.

A sanity example with both quantities nonzero: for constant predictors on $y\sim\mathcal N(\mu,\sigma^2)$, $\hat\theta_n=\bar y$, $\mathbb E\,\mathcal E_n=\sigma^2/n$ and $\mathbb E\,\mathrm{gap}_n=2\sigma^2/n$.
"""
    ),
    markdown(
        r"""
<a id="wb2-section-1b"></a>

### 1.B. Ordinary least squares, exactly

**The question.** How large are the gap and the excess risk in the simplest
model where everything can be computed? Least squares is that model, and it is
the reference case for most of this workbook: it gives the "parameter
counting" answer, about $\sigma^2/n$ of excess risk per parameter, and shows
where it breaks down as $p$ approaches $n$.

**Setup.** The linear model is $y=x^\top\beta+\varepsilon$, with
$x,\beta\in\mathbb R^p$ and noise $\varepsilon$ of mean $0$ and variance
$\sigma^2$, independent of $x$. Stack the $n$ training inputs as the rows of the
*design matrix* $X\in\mathbb R^{n\times p}$ and the targets as
$y\in\mathbb R^n$, so $y=X\beta+\varepsilon$ with $\varepsilon\in\mathbb R^n$.
Here take $\varepsilon\sim\mathcal N(0,\sigma^2I_n)$, $n>p$, and $X$ of full
column rank (its columns are linearly independent). Ordinary least squares
(OLS) is ERM with the squared loss, $\hat R_n(b)=\frac1n\|y-Xb\|^2$. For the
population, take a fresh $x\sim\mathcal N(0,I_p)$ with $y=x^\top\beta+\varepsilon$.

**A tool used throughout: the SVD.** Any $X$ has a thin singular value
decomposition $X=U\,\mathrm{diag}(s_1,\dots,s_r)\,V^\top$, where $r$ is the rank,
the columns $u_k\in\mathbb R^n$ of $U$ and $v_k\in\mathbb R^p$ of $V$ are
orthonormal, and the singular values $s_k>0$. Then
$X^\top X=\sum_ks_k^2v_kv_k^\top$: the $v_k$ are the principal directions of
the inputs, and $s_k^2$ measures how much the data vary along $v_k$ (how
strongly the data "see" that direction). Directions with small $s_k$ are
poorly determined by the data. Most results in 1.B–1.D are statements about
what happens direction by direction in this basis.

*Source:* least squares and the optimism of the training error are in
[*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/)
(Hastie, Tibshirani and Friedman), §3.2 and §7.4.

#### 1.B.1. Mean, covariance and excess risk at a fixed design

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
#### 1.B.2. Random design: parameter counting

Now let the rows of $X$ be i.i.d. $\mathcal N(0,I_p)$, so $X^\top X$ is Wishart.

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

*The constant.* By the Schur complement, $[(X^\top X)^{-1}]_{11}=1/\big(x_{\cdot1}^\top(I-P_{-1})x_{\cdot1}\big)$. Conditional on the other columns, $I-P_{-1}$ is an orthogonal projection of rank $n-p+1$, and it is independent of $x_{\cdot1}\sim\mathcal N(0,I_n)$. So the denominator is $\chi^2_{n-p+1}$. Next, $\mathbb E[1/\chi^2_k]=\frac{\Gamma(k/2-1)}{2\Gamma(k/2)}=\frac1{k-2}$ for $k>2$, which gives $c=1/(n-p-1)$. Combining with 1.B.1.2,

$$
\mathbb E\,\mathcal E_n=\sigma^2\,\mathrm{Tr}\,\mathbb E(X^\top X)^{-1}=\frac{\sigma^2p}{n-p-1}.
$$

**2.** $\mathbb E R(\hat\beta)=\sigma^2+\sigma^2p/(n-p-1)=\sigma^2\frac{n-1}{n-p-1}$, and by 1.B.1.3, $\mathbb E\hat R_n(\hat\beta)=\sigma^2\frac{n-p}{n}$. So

$$
\mathbb E\,\mathrm{gap}_n=\sigma^2\Big[\frac{n-1}{n-p-1}-\frac{n-p}{n}\Big]
\approx\sigma^2\Big[\Big(1+\frac pn\Big)-\Big(1-\frac pn\Big)\Big]=\frac{2\sigma^2p}{n}.
$$

The test loss *overestimates* $\sigma^2$ by about $\sigma^2p/n$ (the excess risk), and the training loss *underestimates* it by the same amount, because the fit has absorbed the noise along $p$ directions. The gap counts both. This is the factor of two between "optimism" and "excess risk" that 4.A.2 derives in general (TIC vs $\tfrac1{2n}\mathrm{Tr}(H^{-1}C)$).

**3.** Each of the $p$ coefficients is estimated with variance $\approx\sigma^2/n$, which is "parameter counting". The exact correction $n/(n-p-1)$ diverges as $p\to n-1$. The cause is the smallest eigenvalues of $X^\top X/n$. When $p/n\to\gamma<1$, its spectrum fills $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (Marchenko–Pastur, quoted), and $\mathrm{Tr}(X^\top X)^{-1}=\frac1n\sum_i 1/\lambda_i(X^\top X/n)$ is dominated by the eigenvalues near the lower edge, which tends to $0$ as $\gamma\to1$. Noise along nearly degenerate directions of the design is amplified without limit. §3 (3.A) picks this up.

(All of these formulas are checked by Monte Carlo in `checks/wb2a_check.py`.)
"""
    ),
    markdown(
        r"""
When someone says deep nets "generalize mysteriously", ask which quantity they
mean. Zero training loss with a small test loss is a statement about the gap
(and the test loss). A statement that more parameters do not hurt is about the
excess risk, or about the test loss as a function of $p$.
"""
    ),
    # ------------------------------------------------------------------ §1.C
    markdown(
        r"""
<a id="wb2-section-1c"></a>

### 1.C. Bias–variance

**The question.** Why can a model that fits its training data worse predict
better? The classical answer: a learned predictor errs on new data for three
separate reasons. The target is noisy (*noise*). The learning rule may be
systematically off, on average over training sets (*bias*). And the prediction
changes from one training set to another (*variance*). Flexible models have low
bias and high variance; constrained models the reverse. Regularization (1.D)
deliberately buys a little bias to remove a lot of variance.

**What you get.** 1.C.1 derives the exact decomposition for squared loss,
shows that the same algebra compares a single posterior draw with the
posterior average, and shows that it fails for 0–1 loss. 1.C.2 does ridge
regression direction by direction: which directions carry the variance, why
some ridge always beats none, and how to count "effective parameters".

*Source:* ESL ([*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/),
Hastie, Tibshirani and Friedman) §7.3 (the decomposition) and §3.4.1 (ridge in
the singular-value basis, and its effective degrees of freedom).

#### 1.C.1. Decompose, then decompose a posterior

Let $y=f(x)+\varepsilon$, where $f$ is the true regression function,
$\mathbb E[\varepsilon\mid x]=0$ and $\mathrm{Var}[\varepsilon\mid x]=\sigma^2(x)$.
A predictor $\hat f_S$ is trained on a random sample $S$, independent of the
test point. Fix a test input $x$; expectations are over $S$ and the test noise.

1. Show that $\mathbb E_{S,\varepsilon}\big(\hat f_S(x)-y\big)^2=\sigma^2(x)+\big(\mathbb E_S\hat f_S(x)-f(x)\big)^2+\mathrm{Var}_S\hat f_S(x)$.
2. Average $K$ predictors trained on independent samples. Which terms change, and by how much?
3. The same algebra with a different source of randomness. Let $q$ be any distribution over parameters, for instance a posterior given $S$, with predictive mean $\bar f(x)=\mathbb E_{\theta\sim q}f_\theta(x)$. Show that for squared loss
$$\mathbb E_{\theta\sim q}\big(f_\theta(x)-y\big)^2=\big(\bar f(x)-y\big)^2+\mathrm{Var}_{\theta\sim q}f_\theta(x).$$
Translate this into a statement comparing a single posterior draw with the posterior-predictive mean. (4.A.6 calls their errors the Gibbs and Bayes errors.)
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

with equality iff the posterior predictors agree for almost every input under the test distribution. For squared loss, ensembling never hurts a posterior draw. 4.A.6 finds that in the regular case the excess parts differ by a factor of two ($p/n$ vs $p/2n$).

**4.** No: the 0–1 loss does not decompose additively. Take $y=1$ deterministically and predict $\mathbf 1[\hat g_S(x)>0]$, where $\hat g_S(x)\sim\mathcal N(\mu,v)$ with $\mu<0$, so the average prediction is on the wrong side. The error is $\Pr[\hat g_S\le0]=\Phi(-\mu/\sqrt v)$, which *decreases* toward $\tfrac12$ as $v$ grows. When the bias has the wrong sign, variance helps. "Variance always hurts" is a fact about squared loss.
"""
    ),
    markdown(
        r"""
#### 1.C.2. Ridge in the eigenbasis: bias and variance per direction

Take the linear model $y=X\beta+\varepsilon$ with a fixed $n\times p$ design
$X$ and noise $\varepsilon\sim(0,\sigma^2I_n)$, $\sigma^2>0$. For the
comparisons with OLS below assume $X$ has full column rank; the positive-penalty
formulas also apply to rank-deficient designs. *Ridge regression* adds a
squared-norm penalty to least squares:

$$
\hat\beta_\lambda=\arg\min_b\ \tfrac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y .
$$

Write $X^\top X=V\,\mathrm{diag}(s_1^2,\dots,s_p^2)\,V^\top$, where the $s_k$ are
the singular values of $X$ and the $v_k$ (columns of $V$) the principal
directions of the inputs, and write $\beta_k=v_k^\top\beta$ for the true
coefficient along $v_k$. Write $a=n\lambda$ for short. We track the parameter
error $\mathbb E\|\hat\beta_\lambda-\beta\|^2$, which equals the excess
population risk when test inputs are $x\sim\mathcal N(0,I)$ (1.B.1.2).

1. Show that, in coordinates $v_k^\top\hat\beta_\lambda$,
$$\mathrm{bias}_k=-\frac{a}{s_k^2+a}\,\beta_k,\qquad \mathrm{var}_k=\sigma^2\frac{s_k^2}{(s_k^2+a)^2},$$
and that different coordinates are uncorrelated. Hence $\mathbb E\|\hat\beta_\lambda-\beta\|^2=\sum_k\dfrac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$.
2. Which directions carry most of the variance at $\lambda=0$? Show that the derivative of the total error with respect to $a$ at $a=0$ is strictly negative, so some ridge always beats OLS.
3. Allow a separate penalty $a_k$ in each direction. Show that the optimum is $a_k=\sigma^2/\beta_k^2$, independent of $s_k$, and that the minimal error in that direction is $\big(s_k^2/\sigma^2+1/\beta_k^2\big)^{-1}$. If $\beta\sim\mathcal N(0,\tau^2I)$, show that the best *common* penalty is $a=\sigma^2/\tau^2$.
4. *Effective degrees of freedom.* The fitted values are $\hat y=S_\lambda y$ with $S_\lambda=X(X^\top X+aI)^{-1}X^\top$. Show that $\mathrm{Tr}\,S_\lambda=\mathrm{df}(\lambda)=\sum_k s_k^2/(s_k^2+a)$. Then show that the in-sample optimism, $\mathbb E\frac1n\|y'-\hat y\|^2-\mathbb E\frac1n\|y-\hat y\|^2$ for fresh noise $y'=X\beta+\varepsilon'$ at the same inputs, equals $2\sigma^2\,\mathrm{df}(\lambda)/n$. Compare with 1.B.2.

<details><summary>Hint</summary>

For 1, $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, and the covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}$. All of these matrices are diagonal in $V$. For 3, differentiate $f(a)=(a^2\beta^2+\sigma^2s^2)/(s^2+a)^2$; the numerator of $f'$ factors. For 4, expand both squared norms around $X\beta$. The only surviving difference is $\mathbb E\langle\varepsilon,S_\lambda\varepsilon\rangle$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, so in the $V$ basis coordinate $k$ is $\frac{s_k^2}{s_k^2+a}\beta_k$ and the bias is $\big(\frac{s_k^2}{s_k^2+a}-1\big)\beta_k=-\frac{a}{s_k^2+a}\beta_k$. The covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}=V\,\mathrm{diag}\!\big(\sigma^2s_k^2/(s_k^2+a)^2\big)V^\top$, which is diagonal in $V$, so the coordinates are uncorrelated. Summing squared bias and variance over $k$ gives the total.

**2.** At $a=0$, $\mathrm{var}_k=\sigma^2/s_k^2$, so the weakest directions (small $s_k$) dominate. That is the same $\mathrm{Tr}(X^\top X)^{-1}$ as in 1.B.1.2. Write $f_k(a)=\frac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$. Then

$$
f_k'(a)=\frac{2a\beta_k^2(s_k^2+a)-2(a^2\beta_k^2+\sigma^2s_k^2)}{(s_k^2+a)^3}=\frac{2s_k^2(a\beta_k^2-\sigma^2)}{(s_k^2+a)^3}.
$$

At $a=0$ this is $-2\sigma^2/s_k^4<0$ for every $k$, so a small amount of ridge lowers the total error, for every $\beta$. This is the observation of [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634).

**3.** By the factored derivative, $f_k$ decreases for $a<\sigma^2/\beta_k^2$ and increases after, so $a_k^*=\sigma^2/\beta_k^2$, which does not involve $s_k$. Substituting,

$$
f_k(a_k^*)=\frac{\sigma^2(\sigma^2+s_k^2\beta_k^2)/\beta_k^2}{(s_k^2\beta_k^2+\sigma^2)^2/\beta_k^4}=\frac{\sigma^2\beta_k^2}{s_k^2\beta_k^2+\sigma^2}=\Big(\frac{s_k^2}{\sigma^2}+\frac1{\beta_k^2}\Big)^{-1}.
$$

If $\beta_k=0$, the optimum is the limit $a_k\to\infty$, with zero error. This has the form of a posterior variance: data precision plus prior precision. If $\beta\sim\mathcal N(0,\tau^2I)$, then $\mathbb E\beta_k^2=\tau^2$, and $\mathbb E_\beta f_k(a)=\frac{a^2\tau^2+\sigma^2s_k^2}{(s_k^2+a)^2}$ is minimized at $a=\sigma^2/\tau^2$ for every $k$ at once. So the best common penalty is $n\lambda=\sigma^2/\tau^2$. 1.D.1 gets the same value as the MAP prior, and 3.C.1.2 as the optimal $\lambda^*=\sigma^2\gamma/r^2$ for random designs, where $\gamma=p/n$, $r^2=\|\beta\|^2$ and $\tau^2=r^2/p$.

**4.** With the thin SVD $X=U\,\mathrm{diag}(s_k)V^\top$, $S_\lambda=U\,\mathrm{diag}\big(s_k^2/(s_k^2+a)\big)U^\top$, so $\mathrm{Tr}\,S_\lambda=\sum_k s_k^2/(s_k^2+a)\in[0,p]$. Let $\mu=X\beta$. Then

$$
\|y'-\hat y\|^2=\|\varepsilon'\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon',\mu-\hat y\rangle,\qquad
\|y-\hat y\|^2=\|\varepsilon\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon,\mu-\hat y\rangle .
$$

The first cross term has mean $0$ because $\varepsilon'$ is independent of $\hat y$. In the second, $\mu-\hat y=(I-S_\lambda)\mu-S_\lambda\varepsilon$, so $\mathbb E\langle\varepsilon,\mu-\hat y\rangle=-\sigma^2\,\mathrm{Tr}\,S_\lambda$. The difference is $2\sigma^2\,\mathrm{Tr}\,S_\lambda$, and dividing by $n$ gives $2\sigma^2\mathrm{df}(\lambda)/n$. At $\lambda=0$, $\mathrm{df}=p$, which has the same leading $2\sigma^2p/n$ value as 1.B.2. Here it is exact for new labels at the training inputs; 1.B.2 evaluates fresh random inputs, so its finite-sample gap differs. Ridge replaces "count the parameters" with "count the directions the data can resolve against the penalty."
"""
    ),
    markdown(
        r"""
The textbook U-shaped curve, where test error falls with model size and then
rises, is 1.C.2 with the penalty replaced by the number of features. §3 shows
the U is not the whole story even in least squares.
"""
    ),
]

REG_PRIOR = [
    # ------------------------------------------------------------------ §1.D
    markdown(
        r"""
#### 1.D.1. Ridge as a prior: MAP, posterior, and a warning

**Why.** "Weight decay is a Gaussian prior" is the most common justification
of the most common regularizer. This exercise checks the claim exactly in
linear regression and finds where it stops being a statement about the
learned function.

**Setup.** The linear model $y=X\beta+\varepsilon$ with a fixed design
$X\in\mathbb R^{n\times p}$ and noise $\varepsilon\sim\mathcal N(0,\sigma^2I_n)$;
ridge regression is $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$, the
minimizer of $\frac1n\|y-Xb\|^2+\lambda\|b\|^2$. *MAP* (maximum a
posteriori) is the mode of the posterior density.

*Source:* ESL §3.4.1 notes that ridge is the posterior mode and mean under a
Gaussian prior.

1. Take the prior $\beta\sim\mathcal N(0,\tau^2I)$ and the likelihood $y\mid\beta\sim\mathcal N(X\beta,\sigma^2I)$. Show that the MAP estimate is ridge with $\lambda=\sigma^2/(n\tau^2)$. Show that the posterior is $\mathcal N\big(\hat\beta_\lambda,\ \sigma^2(X^\top X+n\lambda I)^{-1}\big)$, so the posterior mean equals the MAP here.
2. Compare the posterior covariance with the sampling covariance of $\hat\beta_\lambda$ over the noise at fixed $\beta$, which is $\sigma^2(X^\top X+n\lambda I)^{-1}X^\top X(X^\top X+n\lambda I)^{-1}$ (1.C.2). Which is larger, and why do they differ?
3. MAP depends on the parametrization. Suppose a posterior on $\beta>0$ has density $\pi(\beta)\propto\beta e^{-\beta}$. Find its mode. Now write $\beta=e^u$, find the mode of the density of $u$, and map it back. Which of the mode, mean and median is invariant under monotone reparametrization? What does this mean for "weight decay is a prior"?
"""
    ),
    answer(
        r"""
**1.** The negative log posterior is $\frac1{2\sigma^2}\|y-X\beta\|^2+\frac1{2\tau^2}\|\beta\|^2+\text{const}$. Multiplying by $2\sigma^2/n$ gives $\frac1n\|y-X\beta\|^2+\frac{\sigma^2}{n\tau^2}\|\beta\|^2$, the ridge objective with $\lambda=\sigma^2/(n\tau^2)$. The log posterior is quadratic in $\beta$, so the posterior is Gaussian. Its precision is $X^\top X/\sigma^2+I/\tau^2=(X^\top X+n\lambda I)/\sigma^2$, and its mean (which is also the mode) is $\hat\beta_\lambda$.

**2.** Write $G=X^\top X$ and $a=n\lambda$. The posterior covariance is $\sigma^2(G+aI)^{-1}$ and the sampling covariance is $\sigma^2G(G+aI)^{-2}$. Their difference is $\sigma^2a(G+aI)^{-2}\succeq0$, so the posterior is wider. The sampling covariance ignores the bias. The posterior covariance is the bias–variance mean squared error averaged over the prior: along a singular direction $v_k$ of $X$ with singular value $s$ (and $\mathbb E\beta_k^2=\tau^2$ under the prior), using 1.C.2.1 at $a=\sigma^2/\tau^2$,

$$
\frac{a^2\tau^2+\sigma^2s^2}{(s^2+a)^2}=\frac{\sigma^2(a+s^2)}{(s^2+a)^2}=\frac{\sigma^2}{s^2+a}.
$$

**3.** $\frac{d}{d\beta}\log(\beta e^{-\beta})=1/\beta-1=0$, so the mode is $\beta=1$. The density of $u=\log\beta$ is $\pi(e^u)e^u\propto e^{2u-e^u}$, and $\frac{d}{du}(2u-e^u)=0$ gives $e^u=2$, i.e. $\beta=2$. Same posterior, different "MAP". The mean is not invariant under nonlinear maps either. The median is invariant under monotone maps.

So the identification "penalty = negative log prior" holds *in a chosen parametrization*. Weight decay is a Gaussian prior on the weights, which in a network induces some prior on functions that depends on the architecture. The MAP estimate under it is not an invariant summary of the posterior.
"""
    ),
    markdown(
        r"""
#### 1.D.2. Ridge as a constraint: the Lagrange multiplier

**Why.** The second reading of a penalty: minimize the training loss over a
ball $\|b\|^2\le c$ of allowed parameters. Small balls contain few functions,
which is the "capacity" explanation of why regularized models generalize.
This exercise shows the penalty and the ball are the same estimator, and asks
what the capacity explanation then predicts.

Take the linear model of 1.D, with design $X\in\mathbb R^{n\times p}$ and targets $y\in\mathbb R^n$, and ridge $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$. $X^+$ is the Moore–Penrose pseudoinverse, so $X^+y$ is the least-squares solution of smallest norm.

1. For $c>0$, show that a minimizer of $\hat R_n(b)=\frac1n\|y-Xb\|^2$ subject to $\|b\|^2\le c$ is $\hat\beta_\lambda$, where $\lambda$ is the KKT (Lagrange) multiplier. Interpret $\hat\beta_0=X^+y$ when the design is rank deficient.
2. Show that $c(\lambda)=\|\hat\beta_\lambda\|^2$ is strictly decreasing when $X^\top y\neq0$. Hence $\lambda(c)$ is well defined for $c<\|X^+y\|^2$, and $\lambda=0$ for $c\ge\|X^+y\|^2$.
3. In this view, what explains why the constrained estimator generalizes? Which objection from §2 (the random-label experiments of 2.D) applies to that explanation?

<details><summary>Hint</summary>

For 1, write the Lagrangian $\hat R_n(b)+\lambda(\|b\|^2-c)$ with $\lambda\ge0$, set its gradient to zero, and use complementary slackness. For 2, write $\|\hat\beta_\lambda\|^2$ in the SVD of $X$ and look at each term as a function of $\lambda$. For 3, recall 1.A.1.3 and §2.C; the Rademacher complexity of a norm ball of linear predictors is in UML §26.2.
</details>
"""
    ),
    answer(
        r"""
**1.** The problem is convex, and Slater's condition holds for $c>0$. The Lagrangian is $\hat R_n(b)+\lambda(\|b\|^2-c)$ with $\lambda\ge0$. Stationarity gives $-\frac2nX^\top(y-Xb)+2\lambda b=0$, i.e. $b=(X^\top X+n\lambda I)^{-1}X^\top y=\hat\beta_\lambda$ for $\lambda>0$. For $\lambda=0$, use the minimum-norm least-squares solution $X^+y$; other feasible least-squares solutions may also minimize when $X$ is rank deficient. Complementary slackness says either $\lambda=0$ (the constraint need not be strictly slack) or $\|\hat\beta_\lambda\|^2=c$.

**2.** In the SVD, $\|\hat\beta_\lambda\|^2=\sum_k\frac{s_k^2(u_k^\top y)^2}{(s_k^2+n\lambda)^2}$. Each term is non-increasing in $\lambda$, and at least one is strictly decreasing when $X^\top y\ne0$. As $\lambda\downarrow0$ the norm rises to $\|X^+y\|^2$, the minimum-norm least-squares solution, and as $\lambda\to\infty$ it falls to $0$. So the map from $c$ to $\lambda$ is a decreasing bijection on $(0,\|X^+y\|^2)$. Tighter constraints need bigger multipliers.

**3.** The explanation is capacity. The norm ball $\{x\mapsto x^\top b:\|b\|\le B\}$ has Rademacher complexity of order $B\sqrt{\mathbb E\|x\|^2/n}$ (quoted), which controls prediction complexity. To turn this into a squared-loss gap bound, additionally control the input norms and labels (or suitable tails), so the loss is Lipschitz on the allowed prediction range. Small enough complexity then gives an excess-risk bound via 1.A.1.3. This inherits two problems. The norm depends on the parametrization (1.D.1.3, 1.D.7), and Zhang et al.'s objection applies: nets with weight decay still fit random labels, so the constraint that is actually active is not small.
"""
    ),
]

REG_FILTER = [
    markdown(
        r"""
#### 1.D.4. Ridge as a spectral filter, and early stopping

**Why.** The third reading, and the most useful for deep learning: ridge acts
on the data's principal directions $v_k$ one at a time, keeping the
well-determined ones (large singular value $s_k$) and shrinking the poorly
determined ones. Gradient descent stopped early does something similar for
free, which is one sense in which "early stopping is regularization". The
exercise compares the two filters exactly and finds the implicit bias of
gradient descent when it runs to convergence.

**Setup.** Design $X\in\mathbb R^{n\times p}$, targets $y\in\mathbb R^n$, ridge
$\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$. The thin SVD is
$X=\sum_ks_ku_kv_k^\top$, with singular values $s_k>0$ and orthonormal
$u_k\in\mathbb R^n$, $v_k\in\mathbb R^p$. A *filter* is a set of factors
$\phi_k\in[0,1]$ saying how much of each direction an estimator keeps.

1. Show that $\hat\beta_\lambda=\sum_{k:\,s_k>0}\phi_k\,\dfrac{u_k^\top y}{s_k}\,v_k$ with $\phi_k^{\mathrm{ridge}}=\dfrac{s_k^2}{s_k^2+n\lambda}$. The factor $(u_k^\top y)/s_k$ is the unregularized coefficient.
2. Run gradient flow $\dot\beta=-\nabla L(\beta)$ on $L(\beta)=\frac1{2n}\|X\beta-y\|^2$, starting from $\beta(0)=0$. Show that $\beta(t)$ has the same form with $\phi_k^{\mathrm{GF}}(t)=1-e^{-s_k^2t/n}$. For gradient descent with step $\eta$ and $T$ steps, show $\phi_k=1-(1-\eta s_k^2/n)^T$.
3. Show that the two filters cross $\tfrac12$ at the same $s^2$ if and only if $t=\ln2/\lambda$. With that matching, write both in terms of $x=s^2/(n\lambda)$, and compare them for $x\ll1$ and for $x\gg1$. Which filter passes more of the weak directions? Which shrinks the strong directions more?
4. Let $p>n$. Show that gradient flow from $\beta(0)=0$ converges to the minimum-norm interpolant $X^+y$. What is the limit from a nonzero initialization $\beta_0$?

<details><summary>Hint</summary>

Work in the coordinates $b_k=v_k^\top\beta$, where $X^\top X$ is diagonal and $X^\top y=\sum_ks_k(u_k^\top y)v_k$; gradient flow then decouples into one scalar linear ODE per $k$. For 3, set each filter equal to $\tfrac12$ and solve for $s^2$. For 4, in which subspace does $\dot\beta$ always lie?
</details>

*Further reading:*
"""
        + reading(
            "A continuous-time view of early stopping for least squares regression",
            "https://arxiv.org/abs/1810.10082",
            "Alnur Ali, J. Zico Kolter, Ryan J. Tibshirani (AISTATS 2019)",
            "Compares the risks of gradient flow and ridge along the whole path, under the "
            r"calibration $t=1/\lambda$.",
            importance=2, length=3,
        )
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

Gradient flow is the sharper, more step-like filter. The largest disagreements are flow above ridge by about $0.138$ (near $x\approx4.3$) and ridge above flow by about $0.044$ (near $x\approx0.36$). With the naive matching $t=1/\lambda$, flow becomes $1-e^{-x}$ and its half-point moves to $x=\ln2$. "Early stopping is ridge" is right in outline (keep the strong directions, drop the weak ones) and wrong in detail. WB5 §1 derives how the polynomial tail of the ridge filter affects learning-curve exponents.

**4.** $\dot\beta\in\mathrm{row}(X)$, so $\beta(t)\in\mathrm{row}(X)$ for all $t$. For $p>n$ with $X$ of full row rank, all $s_k>0$ for $k\le n$, so by 2 the flow converges to $\sum_{k\le n}\frac{u_k^\top y}{s_k}v_k=X^+y$. That is the unique interpolant in the row space, and so the minimum-norm interpolant. From $\beta_0$, the row-space component converges to the same place and the null-space component never moves: the limit is $X^+y+(I-X^+X)\beta_0$. The implicit bias is "minimum distance from the initialization".
"""
    ),
    markdown(
        figure(
            "assets/interactive/spectral_filters.html",
            "Use panel A only (panel B belongs to WB5). Press “match half-points” and read the difference "
            "strip: where do the filters disagree, and which passes more of the weak tail? Then set "
            "t = 1/λ by hand and see the ln 2 offset.",
        )
    ),
]

REG_DROPOUT = [
    markdown(
        r"""
#### 1.D.6. Dropout as adaptive $L^2$ (optional)

**Why.** Dropout randomly zeroes units during training. In a linear model,
averaging over the dropout noise gives an explicit penalty, which shows what
dropout regularizes and in which metric.

**Setup.** Linear predictor $f(x)=\beta^\top x$, training pairs $(x_i,y_i)$
with $x_i\in\mathbb R^p$ the rows of $X\in\mathbb R^{n\times p}$, squared loss.
Drop inputs independently: $\tilde x_{ij}=x_{ij}\xi_{ij}/q$ with
$\xi_{ij}\sim\mathrm{Bernoulli}(q)$, where $q$ is the keep probability. The
$1/q$ keeps $\mathbb E\tilde x_{ij}=x_{ij}$. Part 3 is the main observation of
[Wager, Wang and Liang (2013)](https://arxiv.org/abs/1307.1493).


1. Show that $\mathbb E_\xi\frac1n\sum_i(y_i-\tilde x_i^\top\beta)^2=\frac1n\|y-X\beta\|^2+\frac{1-q}{q}\sum_j\beta_j^2\,\frac{\|x_{\cdot j}\|^2}{n}$, where $x_{\cdot j}$ is the $j$-th column of $X$.
2. Write the penalty as $\beta^\top\Gamma\beta$. Show that it is invariant under rescaling a feature ($x_{\cdot j}\to cx_{\cdot j}$, $\beta_j\to\beta_j/c$), while the ridge penalty is not. Which standard procedure is it equivalent to?
3. Take a generalized linear model with per-example loss $A(x^\top\beta)-y\,x^\top\beta$ (for logistic regression, $A(\eta)=\log(1+e^\eta)$). Expand to second order in the dropout noise and show that the expected penalty is approximately $\frac{1-q}{2q}\cdot\frac1n\sum_iA''(x_i^\top\beta)\sum_jx_{ij}^2\beta_j^2$. In what sense is this "adaptive"?

<details><summary>Hint</summary>

Write $\tilde x_i^\top\beta=x_i^\top\beta+\sum_jx_{ij}\beta_j(\xi_{ij}/q-1)$. The noise has mean zero and $\mathrm{Var}(\xi/q)=(1-q)/q$.
</details>
"""
    ),
    answer(
        r"""
**1.** Let $\eta_i=x_i^\top\beta$ and $\delta_i=\tilde x_i^\top\beta-\eta_i=\sum_jx_{ij}\beta_j(\xi_{ij}/q-1)$. Then $\mathbb E\delta_i=0$ and, by independence, $\mathrm{Var}\,\delta_i=\frac{1-q}{q}\sum_jx_{ij}^2\beta_j^2$. So $\mathbb E(y_i-\eta_i-\delta_i)^2=(y_i-\eta_i)^2+\mathrm{Var}\,\delta_i$. Averaging over $i$ gives the penalty $\frac{1-q}{q}\sum_j\beta_j^2\|x_{\cdot j}\|^2/n$.

**2.** $\Gamma=\frac{1-q}{q}\,\mathrm{diag}(X^\top X/n)$. Under $x_{\cdot j}\to cx_{\cdot j}$, $\beta_j\to\beta_j/c$, the product $\beta_j^2\|x_{\cdot j}\|^2$ is unchanged, while ridge's $\beta_j^2$ changes by $1/c^2$. Dropout on inputs is ridge after standardizing each feature by its root mean square: ridge in the metric $\mathrm{diag}(\hat\Sigma)$.

**3.** $\mathbb E\tilde x_i^\top\beta=\eta_i$, so the linear term $-y\,\tilde x_i^\top\beta$ is unchanged in expectation. A second-order expansion gives $\mathbb E A(\eta_i+\delta_i)\approx A(\eta_i)+\tfrac12A''(\eta_i)\mathrm{Var}\,\delta_i$, so the penalty is $\frac1n\sum_i\tfrac12A''(\eta_i)\frac{1-q}{q}\sum_jx_{ij}^2\beta_j^2$. For logistic regression $A''(\eta)=\hat p(1-\hat p)$, the Fisher information of the example. The penalty is ridge in the metric of the diagonal Fisher information. Confidently classified examples contribute little, and features that matter on uncertain examples are penalized most. "Adaptive" because the metric depends on the current $\beta$. For squared loss written as $\tfrac12(y-\eta)^2$, $A''=1$, which recovers 1 up to the factor $\tfrac12$ of that convention.

The general lesson: many "hacky" regularizers are quadratic penalties in disguise, in some metric. The question to ask is "which metric?"
"""
    ),
]

REG_FLAT = [
    markdown(
        r"""
#### 1.D.7. Why flatness could matter, and why sharpness needs a metric

**Why.** A popular heuristic says that minima in flat regions of the loss
landscape generalize better than sharp ones, and several training methods
(e.g. sharpness-aware minimization) aim for flatness. Here is the exact local
statement behind the heuristic, and the reason the naive version cannot be
right.

**Setup.** Near a minimum $\theta^*$ of the population risk $R$, write
$H=\nabla^2R(\theta^*)$ for the Hessian (the curvature: large eigenvalues are
*sharp* directions, small ones *flat*). A trained parameter
$\hat\theta=\theta^*+\delta$ is random through the training data (and the
optimizer); write $\Sigma=\mathbb E\,\delta\delta^\top$ for its spread around
$\theta^*$. The *per-sample gradient covariance* at the optimum,
$C=\mathrm{Cov}_z[\nabla_\theta\ell(\theta^*;z)]$, measures how much single
examples disagree about which way is downhill; it reappears in §4.

1. Let $\theta^*$ minimize $R$, with $H=\nabla^2R(\theta^*)$, and let the trained parameter be $\hat\theta=\theta^*+\delta$ with $\mathbb E\,\delta\delta^\top=\Sigma$. Show that $\mathbb E R(\hat\theta)-R(\theta^*)\approx\tfrac12\mathrm{Tr}(H\Sigma)$. Check that this is exact for OLS with test inputs $x\sim\mathcal N(0,I_p)$, where $R(b)=\sigma^2+\|b-\beta\|^2$ and $\Sigma=\sigma^2(X^\top X)^{-1}$ (1.B.1).
2. Suppose $\hat\theta$ is ERM in a regular model with $\Sigma\approx\frac1nH^{-1}CH^{-1}$ (the "sandwich" covariance, derived in 4.A.1), where $C$ is the covariance of the per-sample gradients at $\theta^*$. Show that $\tfrac12\mathrm{Tr}(H\Sigma)=\frac1{2n}\mathrm{Tr}(H^{-1}C)$, and that if $C=cH$ this equals $cp/(2n)$, whatever the curvature. What has to be true of $\Sigma$ for "flatter is better" to hold?
3. *Dinh et al.'s rescaling.* Take the two-layer scalar linear network $f(x)=w_2w_1x$ with $R(w)=\mathbb E(y-w_2w_1x)^2$ and $a=\mathbb E x^2>0$. Assume $c^*=\mathbb E[xy]/a\ne0$; the minima are the hyperbola $w_1w_2=c^*$. Compute the Hessian at a minimum, show that $\mathrm{Tr}\,H=2a(w_1^2+w_2^2)$ and that one eigenvalue is zero. Under $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ the function does not change. Show that $\mathrm{Tr}\,H$ takes every value in $[4a|c^*|,\infty)$. Why does the same trick work in deep ReLU networks?
4. Let $\theta=g(\varphi)$ be a smooth reparametrization with Jacobian $J$ at the minimum, and let $\delta$ transform as a tangent vector. Show that $\mathrm{Tr}(H\Sigma)$ is invariant while $\mathrm{Tr}\,H$ is not. What does a meaningful flatness claim have to specify?

<details><summary>Hint</summary>

For 1, the gradient vanishes at $\theta^*$, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\,\mathbb E\,\delta\delta^\top)$. For 3, write $R=a(w_1w_2-c^*)^2+\text{const}$. For 4, at a critical point the Hessian transforms as $J^\top HJ$, because the term involving the second derivative of $g$ multiplies $\nabla R=0$.
</details>
"""
    ),
    answer(
        r"""
**1.** $R(\theta^*+\delta)=R(\theta^*)+\nabla R(\theta^*)^\top\delta+\tfrac12\delta^\top H\delta+O(\|\delta\|^3)$. The gradient term vanishes, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\Sigma)$. Here $\Sigma$ is the second moment about $\theta^*$, so it includes any bias. For OLS, $R(b)=\sigma^2+\|b-\beta\|^2$ is exactly quadratic with $H=2I$, and $\Sigma=\sigma^2(X^\top X)^{-1}$, so $\tfrac12\mathrm{Tr}(H\Sigma)=\sigma^2\mathrm{Tr}(X^\top X)^{-1}$ exactly, as in 1.B.1.2.

**2.** $\tfrac12\mathrm{Tr}\big(H\cdot\tfrac1nH^{-1}CH^{-1}\big)=\tfrac1{2n}\mathrm{Tr}(CH^{-1})$. If $C=cH$, this is $cp/(2n)$. The curvature cancels: a sharp direction is pinned down more tightly by the data ($\Sigma\propto H^{-1}$), and this exactly offsets the larger cost of a given error there. "Flatter is better" needs $\Sigma$ that does not shrink in proportion as $H$ grows. One case is a perturbation scale set by something other than the data's curvature, such as SGD noise at a fixed learning rate: if the gradient noise scales as $C\propto H^{g}$, 4.B.3 finds $\Sigma\propto H^{g-1}$ (it writes the exponent as $\gamma$). So "flat minima generalize" is a claim about $H$ *and* $\Sigma$ jointly.

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
#### 1.D.8. Reading: sharp minima can generalize

"""
        + reading(
            "Sharp minima can generalize for deep nets",
            "https://arxiv.org/abs/1703.04933",
            "Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio (ICML 2017)",
            "Because of the rescaling symmetries of ReLU networks, a minimum can be made arbitrarily sharp "
            "under several standard sharpness definitions without changing the function computed. So "
            "flatness in weight space cannot, on its own, explain generalization.",
            "Read §2–§4: the definitions of flatness (volume ε-flatness, Hessian-based measures, "
            "ε-sharpness), the non-negative homogeneity of rectifiers, and the construction you just did "
            "in 1.D.7.3, generalized to deep rectifier networks. §5 (reparametrizing the model and the "
            "input) is short and makes the same point more generally.",
            importance=5, length=3,
        )
        + r"""

**Question.**

Dinh et al. is often cited as "flatness is meaningless". Using 1.D.7.4, say
precisely what the paper shows and what it leaves open. Name one
flatness-like quantity it does not touch.
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
    markdown(
        r"""
**Where this leaves "the hacky bullshit".** In linear models, weight decay,
early stopping, input noise and dropout are all ways of choosing a filter on
the data's principal directions, each in its own metric; L1 is the exception
that selects coordinates instead of shrinking directions. The "prior" reading
and the "capacity constraint" reading of the same penalty make different
predictions about networks trained on random labels (2.D). Flatness enters
through $\Sigma$ and $H$ together, and only once a metric is fixed; PAC-Bayes
(2.E) is one way of fixing it.
"""
    ),
]

