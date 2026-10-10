"""WB2 cells, part A: opening cell and sections 1-5.

Formulas here are checked numerically by checks/wb2a_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    # ------------------------------------------------------------------ opening
    markdown(
        r"""
# Workbook 2 — Statistical learning and regularization: why the hacky bullshit works

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Deep learning practice is full of tricks that look like hacks: weight decay,
early stopping, dropout, "flat minima", training straight through the point
where the training loss hits zero. This workbook derives, in models simple
enough to solve exactly, what each trick does and what it does not. The aim is
that when you meet a claim about generalization, you can say which classical
statement it borrows, and in which regime that statement is a theorem. You do
the derivations. Where a result is quoted rather than derived, the text says so,
and each section links the papers or textbook sections its derivations come from.

**Worked solutions.** Every problem has a worked solution folded directly
after it under "Solution"; click to reveal it when you want it. Solutions are
for everyone. Hints are folded under "Hint".

**Sections.** Choose whatever interests you, in any order. Each section opens
with its result, so you can tell quickly whether you already know it. The
"Uses" column lists what a section actually relies on. The notation block
below covers the shared definitions. §7 and §8 restate what they need; §9's
exercise points back to §4, §7 and §8.

| Section | What you derive or read | Uses |
|---|---|---|
| [1. Risks and gaps](#wb2-section-1) | The generalization gap versus the excess risk; both computed exactly for least squares | — |
| [2. Doesn't some theorem make this impossible? (天无绝人之路)](#wb2-section-2) | No free lunch and an Occam bound; ReLU unit counts and a depth separation; why class-wide capacity bounds are vacuous for real networks. Reading: Zhang et al. (*core*) | §1.A for 2.C; §4 for 2.D.i.2 |
| [3. Bias–variance](#wb2-section-3) | The pointwise decomposition; ridge direction by direction; effective degrees of freedom | §1.B |
| [4. Regularization three ways, and how not to think about it](#wb2-section-4) | Ridge as prior, constraint and spectral filter; early stopping; dropout (optional); when flatness can matter and why raw sharpness cannot. Reading: Dinh et al. (*core*) | §3.B (setup restated; 4.A.i.2 uses 3.B.i.1); §1.B for one check; §1.A.i.3 and §2.C–2.D for 4.B.i.3 |
| [5. Overfitting and double descent](#wb2-section-5) | The least-squares double-descent curve, and why optimal ridge removes the peak | §1.B.ii, §4.A.i.1, §4.C.i.4; §3.A for 5.D.i |
| [6. Information bottleneck and the information plane](#wb2-section-6) | Why mutual information is degenerate for deterministic networks; what binned estimators measure; compression as clustering in a noisy model; the IB as rate–distortion (optional) | — |
| [7. Bayes and MLE generalization error](#wb2-section-7) | The sandwich covariance, TIC, AIC and BIC; optionally, a single fit versus the Bayes ensemble versus a posterior draw | §4.E.i.1 (restated); §1.B for one check; §3–§4 for 7.E.ii |
| [8. Does it cancel? $C$ vs $H$](#wb2-section-8) | Gradient noise versus curvature; the diagonal power law; SGD's stationary covariance versus a posterior | §4.E.i.1, §7.A–7.C (restated) |
| [9. Fantastic generalization measures](#wb2-section-9) | A reading on which published measures predict the gap, with one exercise linking it to §4, §7 and §8 | §4.E.i.3–4, §7 (setup and 7.B), §8.A |

**Notation, used throughout.**

- Data $z=(x,y)\sim P$, and an i.i.d. sample $S=(z_1,\dots,z_n)$.
- A model with parameters $\theta\in\mathbb R^p$ and a loss $\ell(\theta;z)$.
- Population risk $R(\theta)=\mathbb E_{z\sim P}\,\ell(\theta;z)$; empirical risk $\hat R_n(\theta)=\frac1n\sum_{i=1}^n\ell(\theta;z_i)$.
- The empirical risk minimizer (ERM) $\hat\theta_n\in\arg\min_\theta\hat R_n(\theta)$, and the population minimizer $\theta^*\in\arg\min_\theta R(\theta)$. $\theta^*$ is the best the model class can do, not necessarily the truth.
- $n$ is the number of samples and $p$ the number of parameters; §5 uses $\gamma=p/n$.
- The running example is the linear model $y=x^\top\beta+\varepsilon$ with $\mathbb E\varepsilon=0$ and $\mathrm{Var}\,\varepsilon=\sigma^2$. The design matrix is $X\in\mathbb R^{n\times p}$, with rows $x_i^\top$, and $y\in\mathbb R^n$.
"""
    ),
    # ------------------------------------------------------------------ §1
    markdown(
        r"""
<a id="wb2-section-1"></a>

## 1. Risks and gaps

**The result.** The *generalization gap* $R(\hat\theta_n)-\hat R_n(\hat\theta_n)$
and the *excess risk* $R(\hat\theta_n)-R(\theta^*)$ are different quantities.
Either can be large while the other is zero. A held-out set estimates population
risk and hence the gap; excess risk additionally needs the optimum risk. For ERM, the expected gap bounds the expected excess risk. In ordinary
least squares both are exact: the excess risk is $\sigma^2p/(n-p-1)$, about
$\sigma^2/n$ per parameter, and the gap is about twice that.

### 1.A. Two quantities called "generalization"

For any learning rule $S\mapsto\hat\theta_n$, not only ERM, define

$$
\mathrm{gap}_n=R(\hat\theta_n)-\hat R_n(\hat\theta_n),
\qquad
\mathcal E_n=R(\hat\theta_n)-R(\theta^*).
$$

Both are random through $S$. Papers say "generalizes well" for either one,
and sometimes for both in the same paragraph.

#### 1.A.i. Gap without excess, excess without gap

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

Memorizing costs nothing here because the memorized inputs never recur. This is a caricature of benign overfitting; §5 has a real version.

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
### 1.B. Ordinary least squares, exactly

Take $y=X\beta+\varepsilon$ with $\varepsilon\sim\mathcal N(0,\sigma^2I_n)$,
$n>p$, and $X$ of full column rank. Use squared loss, so
$\hat R_n(b)=\frac1n\|y-Xb\|^2$. For the population, take a fresh
$x\sim\mathcal N(0,I_p)$ with $y=x^\top\beta+\varepsilon$.

*Source:* least squares and the optimism of the training error are in
[*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/)
(Hastie, Tibshirani and Friedman), §3.2 and §7.4.

#### 1.B.i. Mean, covariance and excess risk at a fixed design

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
#### 1.B.ii. Random design: parameter counting

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

*The constant.* By the Schur complement, $[(X^\top X)^{-1}]_{11}=1/\big(x_{\cdot1}^\top(I-P_{-1})x_{\cdot1}\big)$. Conditional on the other columns, $I-P_{-1}$ is an orthogonal projection of rank $n-p+1$, and it is independent of $x_{\cdot1}\sim\mathcal N(0,I_n)$. So the denominator is $\chi^2_{n-p+1}$. Next, $\mathbb E[1/\chi^2_k]=\frac{\Gamma(k/2-1)}{2\Gamma(k/2)}=\frac1{k-2}$ for $k>2$, which gives $c=1/(n-p-1)$. Combining with 1.B.i.2,

$$
\mathbb E\,\mathcal E_n=\sigma^2\,\mathrm{Tr}\,\mathbb E(X^\top X)^{-1}=\frac{\sigma^2p}{n-p-1}.
$$

**2.** $\mathbb E R(\hat\beta)=\sigma^2+\sigma^2p/(n-p-1)=\sigma^2\frac{n-1}{n-p-1}$, and by 1.B.i.3, $\mathbb E\hat R_n(\hat\beta)=\sigma^2\frac{n-p}{n}$. So

$$
\mathbb E\,\mathrm{gap}_n=\sigma^2\Big[\frac{n-1}{n-p-1}-\frac{n-p}{n}\Big]
\approx\sigma^2\Big[\Big(1+\frac pn\Big)-\Big(1-\frac pn\Big)\Big]=\frac{2\sigma^2p}{n}.
$$

The test loss *overestimates* $\sigma^2$ by about $\sigma^2p/n$ (the excess risk), and the training loss *underestimates* it by the same amount, because the fit has absorbed the noise along $p$ directions. The gap counts both. This is the factor of two between "optimism" and "excess risk" that §7 derives in general (TIC vs $\tfrac1{2n}\mathrm{Tr}(H^{-1}C)$).

**3.** Each of the $p$ coefficients is estimated with variance $\approx\sigma^2/n$, which is "parameter counting". The exact correction $n/(n-p-1)$ diverges as $p\to n-1$. The cause is the smallest eigenvalues of $X^\top X/n$. When $p/n\to\gamma<1$, its spectrum fills $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (Marchenko–Pastur, quoted), and $\mathrm{Tr}(X^\top X)^{-1}=\frac1n\sum_i 1/\lambda_i(X^\top X/n)$ is dominated by the eigenvalues near the lower edge, which tends to $0$ as $\gamma\to1$. Noise along nearly degenerate directions of the design is amplified without limit. §5 picks this up.

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
    # ------------------------------------------------------------------ §2
    markdown(
        r"""
<a id="wb2-section-2"></a>

## 2. Doesn't some theorem make this impossible? (天无绝人之路)

**The result.** Three classical theorems seem to say that learning from data
should be impossible, or at least unexplained. No free lunch says every learner
is at chance off the training set. Universal approximation says nets can
represent anything, so why this function? The elementary class-wide bounds below are
vacuous at the network sizes considered. Each theorem is true, and none of them bites. NFL
averages over a uniform prior on targets that the world does not have. UAT is
an existence statement, silent on which function is learned or at what cost.
Capacity bounds control the worst function in the class, and Zhang et al.
showed that the class networks actually reach includes memorizers of random
labels. 天无绝人之路, "heaven never closes every road": the way out is a bet on
structure in the targets, made implicitly by the architecture and the training
procedure.

### 2.A. No free lunch

Let the input domain $\mathcal X$ be finite, with $|\mathcal X|=N$, and the
target a binary function $f:\mathcal X\to\{0,1\}$. A training set consists of
inputs $x_1,\dots,x_n$, chosen independently of $f$, with labels $f(x_i)$. Assume at least one input remains unseen. A learner
is any map, possibly randomized, from the labelled sample to a hypothesis
$h:\mathcal X\to\{0,1\}$. With $U$ the set of unseen inputs, the off-training-set
error is
$\mathrm{err}_{\mathrm{OTS}}(h,f)=\frac1{|U|}\sum_{x\in U}\mathbf 1[h(x)\ne f(x)]$.

*Sources:* off-training-set NFL is from
[Wolpert (1996)](https://doi.org/10.1162/neco.1996.8.7.1341). Shalev-Shwartz
and Ben-David's
[*Understanding Machine Learning*](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning)
(free PDF; "UML" below) proves a PAC version of NFL in §5.1, the finite-class
bound of part 3 in §2.3.1, and the description-length (Occam) version in §7.3.

#### 2.A.i. Prove NFL, then find the bet

1. Let $f$ be uniform over all $2^N$ functions. Prove that every learner has expected off-training-set error exactly $\tfrac12$, conditional on any training inputs.
2. Conclude that for any two learners $A$ and $B$, $\sum_f\big[\mathrm{err}_{\mathrm{OTS}}(A,f)-\mathrm{err}_{\mathrm{OTS}}(B,f)\big]=0$. What does this say about a learner that beats another on some targets?
3. Now let $f$ lie in a class $\mathcal F$ with $|\mathcal F|=M$, and let the inputs be i.i.d. from a distribution $D$ on $\mathcal X$. Show that a learner returning any $h\in\mathcal F$ consistent with the sample satisfies $\Pr[\mathrm{err}_D(h)>\epsilon]\le Me^{-n\epsilon}$, where $\mathrm{err}_D$ is the error under $D$. How many samples suffice if $\mathcal F$ is the set of functions computed by programs of at most $b$ bits? What does the bound give for the class of all functions?
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
- a simplicity (Solomonoff) prior, which favours the shortest consistent program (WB4);
- a smoothness prior, which favours kernel or nearest-neighbour methods;
- a symmetry prior, which favours equivariant architectures such as CNNs (WB4).

The question "why does deep learning generalize?" becomes: which prior over targets do the architecture and the optimizer implement, and why does it match the world?
"""
    ),
    markdown(
        r"""
### 2.B. Universal approximation: existence, not mechanism

**UAT (quoted).** For any continuous $f$ on a compact $K\subset\mathbb R^d$ and
any $\epsilon>0$, a one-hidden-layer network with a continuous non-polynomial
activation (for example ReLU) achieves $\sup_K|f-g|<\epsilon$. Cybenko (1989)
and Hornik (1991) proved the sigmoidal versions; the non-polynomial
characterization is
[Leshno, Lin, Pinkus and Schocken (1993)](https://linkinghub.elsevier.com/retrieve/pii/S0893608005801315).
The theorem says nothing about how many units are needed. The problem below
counts them in one dimension. Part 4 follows the depth-separation construction
of [Telgarsky (2015)](https://arxiv.org/abs/1509.08101).

#### 2.B.i. How many units for $\sin\omega x$?

1. Show that a continuous piecewise-linear function on $[0,1]$ with $K$ pieces is exactly a one-hidden-layer ReLU network with $K$ units plus an output bias.
2. *Upper bound.* Show that the piecewise-linear interpolant of $\sin(\omega x)$ at $K+1$ equally spaced knots has sup error at most $\omega^2/(8K^2)$. Deduce that $\lceil\omega/\sqrt{8\epsilon}\,\rceil$ units suffice for accuracy $\epsilon$.
3. *Lower bound.* Show that a one-hidden-layer ReLU network with $m$ units on $\mathbb R$ is piecewise linear with at most $m+1$ pieces. Then show that for $\epsilon<1$ accuracy, $m$ must grow at least like $\omega/(2\pi)$.
4. *Depth.* Let $t(x)=2\,\mathrm{ReLU}(x)-4\,\mathrm{ReLU}(x-\tfrac12)$, the tent map on $[0,1]$. Show that the $k$-fold composition $t^{\circ k}$ has $2^k$ linear pieces. Conclude that a depth-$k$ network with $2k$ units computes a function that any one-hidden-layer network needs at least $2^k-1$ units to represent.
5. List three things UAT does not tell you.

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
### 2.C. Capacity bounds, and why they are vacuous here

§1.A.i.3 reduced the excess risk to the worst gap over the class. For a finite
class with losses in $[0,1]$, Hoeffding's inequality and a union bound give,
with probability at least $1-\delta$,

$$
\sup_{h\in\mathcal H}|R(h)-\hat R_n(h)|\le\sqrt{\frac{\ln(2|\mathcal H|/\delta)}{2n}} .
$$

For infinite classes of binary classifiers, $\ln|\mathcal H|$ is replaced by a
combinatorial dimension. When $n$ is sufficiently larger than the VC dimension $d$, the bound has the shape
$\sqrt{(d\ln(n/d)+\ln(1/\delta))/n}$ up to constants (quoted, not proved).
For ReLU networks with $W$ weights in $L$ layers, the general upper bound is
$O(WL\log W)$, with constructions achieving $\Omega(WL\log(W/L))$;
this does not assign that dimension to every architecture. See
[Bartlett et al. (2019)](https://jmlr.org/papers/v20/17-612.html). Finite
classes and VC dimension are in UML ch. 4 and ch. 6.

#### 2.C.i. Plug in a real network

1. Derive the finite-class bound from Hoeffding's inequality, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$, and a union bound.
2. A network with $W=10^6$ float32 weights is a finite class. Evaluate the bound for $n=5\times10^4$ (the size of the CIFAR-10 training set) and $\delta=0.05$. Then compare $d\approx W\ln W$ with $n$.
3. Zhang et al. (below) show that this kind of architecture, trained by SGD, fits CIFAR-10 images with uniformly random labels to zero training error. Let $\mathcal H$ be the set of functions the architecture and SGD can produce. Assume such interpolation succeeds with high probability on i.i.d. samples from a distribution with independent uniform labels and distinct inputs. Show that every bound of the form "for every distribution $P$, with probability $\ge1-\delta$, all $h\in\mathcal H$ have $\mathrm{gap}(h)\le\epsilon(n,\delta)$" must have $\epsilon\ge0.9$ (0–1 loss, 10 classes), up to sampling fluctuation.

<details><summary>Hint</summary>

For 3, the bound has to hold for *every* $P$. Choose one.
</details>
"""
    ),
    answer(
        r"""
**1.** For a fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$. A union bound over $|\mathcal H|$ hypotheses gives $\Pr[\sup_h|\cdot|>t]\le2|\mathcal H|e^{-2nt^2}$. Setting this equal to $\delta$ and solving gives $t=\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.

**2.** Using the parameter-encoding upper bound $\ln|\mathcal H|\le32W\ln2\approx2.22\times10^7$, so the bound is $\sqrt{(2.22\times10^7+\ln 40)/10^5}\approx\sqrt{222}\approx14.9$. Since the 0–1 loss is at most 1, this parameter-encoding bound says nothing. The rough VC scale is $W\ln W\approx1.4\times10^7\gg n=5\times10^4$. The VC route needs $n\gg d$, so a dimension of that size would make it uninformative too. This scale comes from the generic upper bound; it does not prove that the actual architecture has that VC dimension. If the actual VC dimension is at least $n$, the class shatters some $n$ points. Under uniform sampling on those points, a random target leaves an expected unseen mass $(1-1/n)^n$, and NFL gives expected error at least $\tfrac12(1-1/n)^n$ for some target. Thus uniformly small error is impossible at that sample size; this is not a claim that every distribution-free bound is impossible.

**3.** Take $P'$ to be CIFAR-10 images with labels drawn uniformly from the 10 classes, independently of the image. Assume the random-label fitting result extends to i.i.d. samples from this input distribution, with distinct inputs and high probability of interpolation. Then SGD reaches an $h\in\mathcal H$ with $\hat R_n(h)=0$. Its labels are independent of the input, so $R(h)=0.9$. So on $P'$, some $h\in\mathcal H$ has gap $0.9$ with high probability. A bound that depends only on $\mathcal H$, $n$ and $\delta$ must hold for $P'$, so $\epsilon(n,\delta)\ge0.9-o(1)$. The same distribution-independent bound is then vacuous for the real labels as well. The experiment on one fixed image collection motivates the interpolation assumption; it does not prove it for every random draw. Vacuous is not wrong, but it cannot be the explanation. 2.D.i.1 asks which explanations survive.
"""
    ),
    markdown(
        r"""
### 2.D. Reading

"""
        + reading(
            "Understanding deep learning requires rethinking generalization",
            "https://arxiv.org/abs/1611.03530",
            "Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals (ICLR 2017)",
            "Standard image-classification networks trained by SGD fit random labels, and even random "
            "pixels, to zero training error, with or without explicit regularization. So capacity "
            "measures of the model class cannot explain why the same networks generalize on real labels.",
            "Read §1 (the claim and the contributions) and §2 (the randomization tests: true labels, "
            "partially and fully random labels, shuffled and random pixels; how training time changes). "
            "§3 (the role of explicit and implicit regularization) is short and worth skimming. The "
            "finite-sample expressivity construction (§4) and the linear-model discussion (§5) are optional.",
            tier="core",
        )
        + r"""

#### 2.D.i. Questions on Zhang et al.

1. List the explanations of generalization that the random-labels experiment rules out, and those it leaves standing.
2. The paper finds that explicit regularizers (weight decay, dropout, data augmentation) help but are neither necessary nor sufficient. Which of §4's three views of regularization (prior, constraint, spectral filter) does this count against, and which survive? (This uses §4.)
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
- norm-, margin-, compression- and PAC-Bayes-style bounds evaluated on the learned function;
- a prior over functions whose support is everything but whose mass is on structured functions.

The slower fit on random labels is a hint that some simplicity ordering is at work. Such a "soft" bias (the class can fit anything, but training prefers some fits) predicts things a hard capacity restriction does not:
- the same model at the same $n$ generalizes differently depending on the labels;
- random labels take longer to fit and end at larger norm or complexity;
- adding parameters need not hurt, and can help (§5), whereas a class-wide capacity upper bound can worsen with class size without predicting that actual risk must worsen;
- a bound in terms of the prior mass or description length of the *found* function can be non-vacuous where class-capacity bounds are not.

**2.** It counts against the *constraint* view used as an explanation, in which a norm ball is small enough for its capacity to explain generalization. The nets generalize without the constraint, and with the constraint they still fit random labels, so the constrained class is still huge. The *prior* view survives if the prior is soft and implemented implicitly, by architecture, initialization and SGD. The *spectral-filter* view also survives as implicit regularization: early stopping and gradient descent's preference for large-eigenvalue directions, and gradient descent's convergence to the minimum-norm solution (§4.C).
"""
    ),
    # ------------------------------------------------------------------ §3
    markdown(
        r"""
<a id="wb2-section-3"></a>

## 3. Bias–variance

**The result.** For squared loss, the expected error at a point splits exactly
into noise, squared bias and variance. The same algebra shows that a single
random member of an ensemble, or a single posterior draw, does worse than the
average prediction by the spread of the members' predictions. Ridge regression makes the bias–variance
split explicit in the eigenbasis of $X^\top X$. Direction $k$ is shrunk by
$s_k^2/(s_k^2+n\lambda)$, which trades variance in weak directions for bias.
Some ridge always beats none, and the best penalty in direction $k$ is
$n\lambda_k=\sigma^2/\beta_k^2$, whatever $s_k$ is. The effective number of
parameters is $\sum_k s_k^2/(s_k^2+n\lambda)$.

### 3.A. The pointwise decomposition

Let $y=f(x)+\varepsilon$ with $\mathbb E[\varepsilon\mid x]=0$ and
$\mathrm{Var}[\varepsilon\mid x]=\sigma^2(x)$. An estimator $\hat f_S$ is trained
on a random sample $S$, independent of the test point. Fix $x$.

*Source:* ESL §7.3 (the decomposition) and §3.4.1 (ridge in the singular-value
basis, and its effective degrees of freedom).

#### 3.A.i. Decompose, then decompose a posterior

1. Show that $\mathbb E_{S,\varepsilon}\big(\hat f_S(x)-y\big)^2=\sigma^2(x)+\big(\mathbb E_S\hat f_S(x)-f(x)\big)^2+\mathrm{Var}_S\hat f_S(x)$.
2. Average $K$ predictors trained on independent samples. Which terms change, and by how much?
3. The same algebra with a different source of randomness. Let $q$ be any distribution over parameters, for instance a posterior given $S$, with predictive mean $\bar f(x)=\mathbb E_{\theta\sim q}f_\theta(x)$. Show that for squared loss
$$\mathbb E_{\theta\sim q}\big(f_\theta(x)-y\big)^2=\big(\bar f(x)-y\big)^2+\mathrm{Var}_{\theta\sim q}f_\theta(x).$$
Translate this into a statement comparing a single posterior draw with the posterior-predictive mean. (§7.E calls their errors the Gibbs and Bayes errors.)
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

with equality iff the posterior predictors agree for almost every input under the test distribution. For squared loss, ensembling never hurts a posterior draw. §7 finds that in the regular case the excess parts differ by a factor of two ($p/n$ vs $p/2n$).

**4.** No: the 0–1 loss does not decompose additively. Take $y=1$ deterministically and predict $\mathbf 1[\hat g_S(x)>0]$, where $\hat g_S(x)\sim\mathcal N(\mu,v)$ with $\mu<0$, so the average prediction is on the wrong side. The error is $\Pr[\hat g_S\le0]=\Phi(-\mu/\sqrt v)$, which *decreases* toward $\tfrac12$ as $v$ grows. When the bias has the wrong sign, variance helps. "Variance always hurts" is a fact about squared loss.
"""
    ),
    markdown(
        r"""
### 3.B. Ridge in the eigenbasis

Use a fixed design and noise $\varepsilon\sim(0,\sigma^2I_n)$ and $\sigma^2>0$. For the comparisons with OLS below assume $X$ has full column rank; the positive-penalty formulas also apply to rank-deficient designs. Ridge regression is

$$
\hat\beta_\lambda=\arg\min_b\ \tfrac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y .
$$

Write $X^\top X=V\,\mathrm{diag}(s_1^2,\dots,s_p^2)\,V^\top$, where the $s_k$ are
the singular values of $X$, and $\beta_k=v_k^\top\beta$. Write $a=n\lambda$ for
short. We track the parameter error $\mathbb E\|\hat\beta_\lambda-\beta\|^2$,
which equals the excess population risk when $x\sim\mathcal N(0,I)$ (§1.B).

#### 3.B.i. Bias and variance per direction

1. Show that, in coordinates $v_k^\top\hat\beta_\lambda$,
$$\mathrm{bias}_k=-\frac{a}{s_k^2+a}\,\beta_k,\qquad \mathrm{var}_k=\sigma^2\frac{s_k^2}{(s_k^2+a)^2},$$
and that different coordinates are uncorrelated. Hence $\mathbb E\|\hat\beta_\lambda-\beta\|^2=\sum_k\dfrac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$.
2. Which directions carry most of the variance at $\lambda=0$? Show that the derivative of the total error with respect to $a$ at $a=0$ is strictly negative, so some ridge always beats OLS.
3. Allow a separate penalty $a_k$ in each direction. Show that the optimum is $a_k=\sigma^2/\beta_k^2$, independent of $s_k$, and that the minimal error in that direction is $\big(s_k^2/\sigma^2+1/\beta_k^2\big)^{-1}$. If $\beta\sim\mathcal N(0,\tau^2I)$, show that the best *common* penalty is $a=\sigma^2/\tau^2$.
4. *Effective degrees of freedom.* The fitted values are $\hat y=S_\lambda y$ with $S_\lambda=X(X^\top X+aI)^{-1}X^\top$. Show that $\mathrm{Tr}\,S_\lambda=\mathrm{df}(\lambda)=\sum_k s_k^2/(s_k^2+a)$. Then show that the in-sample optimism, $\mathbb E\frac1n\|y'-\hat y\|^2-\mathbb E\frac1n\|y-\hat y\|^2$ for fresh noise $y'=X\beta+\varepsilon'$ at the same inputs, equals $2\sigma^2\,\mathrm{df}(\lambda)/n$. Compare with §1.B.ii.

<details><summary>Hint</summary>

For 1, $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, and the covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}$. All of these matrices are diagonal in $V$. For 3, differentiate $f(a)=(a^2\beta^2+\sigma^2s^2)/(s^2+a)^2$; the numerator of $f'$ factors. For 4, expand both squared norms around $X\beta$. The only surviving difference is $\mathbb E\langle\varepsilon,S_\lambda\varepsilon\rangle$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\mathbb E\hat\beta_\lambda=(X^\top X+aI)^{-1}X^\top X\beta$, so in the $V$ basis coordinate $k$ is $\frac{s_k^2}{s_k^2+a}\beta_k$ and the bias is $\big(\frac{s_k^2}{s_k^2+a}-1\big)\beta_k=-\frac{a}{s_k^2+a}\beta_k$. The covariance is $\sigma^2(X^\top X+aI)^{-1}X^\top X(X^\top X+aI)^{-1}=V\,\mathrm{diag}\!\big(\sigma^2s_k^2/(s_k^2+a)^2\big)V^\top$, which is diagonal in $V$, so the coordinates are uncorrelated. Summing squared bias and variance over $k$ gives the total.

**2.** At $a=0$, $\mathrm{var}_k=\sigma^2/s_k^2$, so the weakest directions (small $s_k$) dominate. That is the same $\mathrm{Tr}(X^\top X)^{-1}$ as in §1.B. Write $f_k(a)=\frac{a^2\beta_k^2+\sigma^2s_k^2}{(s_k^2+a)^2}$. Then

$$
f_k'(a)=\frac{2a\beta_k^2(s_k^2+a)-2(a^2\beta_k^2+\sigma^2s_k^2)}{(s_k^2+a)^3}=\frac{2s_k^2(a\beta_k^2-\sigma^2)}{(s_k^2+a)^3}.
$$

At $a=0$ this is $-2\sigma^2/s_k^4<0$ for every $k$, so a small amount of ridge lowers the total error, for every $\beta$. This is the observation of [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634).

**3.** By the factored derivative, $f_k$ decreases for $a<\sigma^2/\beta_k^2$ and increases after, so $a_k^*=\sigma^2/\beta_k^2$, which does not involve $s_k$. Substituting,

$$
f_k(a_k^*)=\frac{\sigma^2(\sigma^2+s_k^2\beta_k^2)/\beta_k^2}{(s_k^2\beta_k^2+\sigma^2)^2/\beta_k^4}=\frac{\sigma^2\beta_k^2}{s_k^2\beta_k^2+\sigma^2}=\Big(\frac{s_k^2}{\sigma^2}+\frac1{\beta_k^2}\Big)^{-1}.
$$

If $\beta_k=0$, the optimum is the limit $a_k\to\infty$, with zero error. This has the form of a posterior variance: data precision plus prior precision. If $\beta\sim\mathcal N(0,\tau^2I)$, then $\mathbb E\beta_k^2=\tau^2$, and $\mathbb E_\beta f_k(a)=\frac{a^2\tau^2+\sigma^2s_k^2}{(s_k^2+a)^2}$ is minimized at $a=\sigma^2/\tau^2$ for every $k$ at once. So the best common penalty is $n\lambda=\sigma^2/\tau^2$. §4.A gets the same value as the MAP prior, and §5 as the optimal $\lambda^*=\sigma^2\gamma/r^2$, with $\tau^2=r^2/p$.

**4.** With the thin SVD $X=U\,\mathrm{diag}(s_k)V^\top$, $S_\lambda=U\,\mathrm{diag}\big(s_k^2/(s_k^2+a)\big)U^\top$, so $\mathrm{Tr}\,S_\lambda=\sum_k s_k^2/(s_k^2+a)\in[0,p]$. Let $\mu=X\beta$. Then

$$
\|y'-\hat y\|^2=\|\varepsilon'\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon',\mu-\hat y\rangle,\qquad
\|y-\hat y\|^2=\|\varepsilon\|^2+\|\mu-\hat y\|^2+2\langle\varepsilon,\mu-\hat y\rangle .
$$

The first cross term has mean $0$ because $\varepsilon'$ is independent of $\hat y$. In the second, $\mu-\hat y=(I-S_\lambda)\mu-S_\lambda\varepsilon$, so $\mathbb E\langle\varepsilon,\mu-\hat y\rangle=-\sigma^2\,\mathrm{Tr}\,S_\lambda$. The difference is $2\sigma^2\,\mathrm{Tr}\,S_\lambda$, and dividing by $n$ gives $2\sigma^2\mathrm{df}(\lambda)/n$. At $\lambda=0$, $\mathrm{df}=p$, which has the same leading $2\sigma^2p/n$ value as §1.B.ii. Here it is exact for new labels at the training inputs; §1.B.ii evaluates fresh random inputs, so its finite-sample gap differs. Ridge replaces "count the parameters" with "count the directions the data can resolve against the penalty."
"""
    ),
    markdown(
        r"""
The textbook U-shaped curve, where test error falls with model size and then
rises, is §3.B with the penalty replaced by the number of features. §5 shows
the U is not the whole story even in least squares.
"""
    ),
    # ------------------------------------------------------------------ §4
    markdown(
        r"""
<a id="wb2-section-4"></a>

## 4. Regularization three ways, and how not to think about it

**The result.** In linear regression, ridge is three things at once: the MAP
estimate and posterior mean under a Gaussian prior, the Lagrangian form of a
norm constraint, and a filter that shrinks each eigendirection of the data by
$s_k^2/(s_k^2+n\lambda)$. Early-stopped gradient flow is a related filter,
$1-e^{-s_k^2t/n}$, but not the same one: the half-points match at
$t=\ln2/\lambda$, not $1/\lambda$. Dropout is ridge in a data-dependent
diagonal metric. Flatness can matter, because the mean excess risk is locally approximately
$\tfrac12\mathrm{Tr}(H\Sigma)$. But sharpness on its own ($\mathrm{Tr}\,H$) is
not a property of the learned function, so any flatness claim has to say which
metric it uses.

Setup, as in §3.B: $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$,
noise variance $\sigma^2$, and the thin SVD $X=U\,\mathrm{diag}(s_k)\,V^\top$
with left singular vectors $u_k$ and right singular vectors $v_k$.

### 4.A. As a prior

*Source:* ESL §3.4.1 notes that ridge is the posterior mode and mean under a
Gaussian prior.

#### 4.A.i. MAP, posterior, and a warning

1. Take the prior $\beta\sim\mathcal N(0,\tau^2I)$ and the likelihood $y\mid\beta\sim\mathcal N(X\beta,\sigma^2I)$. Show that the MAP estimate is ridge with $\lambda=\sigma^2/(n\tau^2)$. Show that the posterior is $\mathcal N\big(\hat\beta_\lambda,\ \sigma^2(X^\top X+n\lambda I)^{-1}\big)$, so the posterior mean equals the MAP here.
2. Compare the posterior covariance with the sampling covariance of $\hat\beta_\lambda$ from §3.B. Which is larger, and why do they differ?
3. MAP depends on the parametrization. Suppose a posterior on $\beta>0$ has density $\pi(\beta)\propto\beta e^{-\beta}$. Find its mode. Now write $\beta=e^u$, find the mode of the density of $u$, and map it back. Which of the mode, mean and median is invariant under monotone reparametrization? What does this mean for "weight decay is a prior"?
"""
    ),
    answer(
        r"""
**1.** The negative log posterior is $\frac1{2\sigma^2}\|y-X\beta\|^2+\frac1{2\tau^2}\|\beta\|^2+\text{const}$. Multiplying by $2\sigma^2/n$ gives $\frac1n\|y-X\beta\|^2+\frac{\sigma^2}{n\tau^2}\|\beta\|^2$, the ridge objective with $\lambda=\sigma^2/(n\tau^2)$. The log posterior is quadratic in $\beta$, so the posterior is Gaussian. Its precision is $X^\top X/\sigma^2+I/\tau^2=(X^\top X+n\lambda I)/\sigma^2$, and its mean (which is also the mode) is $\hat\beta_\lambda$.

**2.** Write $G=X^\top X$ and $a=n\lambda$. The posterior covariance is $\sigma^2(G+aI)^{-1}$ and the sampling covariance is $\sigma^2G(G+aI)^{-2}$. Their difference is $\sigma^2a(G+aI)^{-2}\succeq0$, so the posterior is wider. The sampling covariance ignores the bias. The posterior covariance is the bias–variance mean squared error averaged over the prior: per direction, at $a=\sigma^2/\tau^2$,

$$
\frac{a^2\tau^2+\sigma^2s^2}{(s^2+a)^2}=\frac{\sigma^2(a+s^2)}{(s^2+a)^2}=\frac{\sigma^2}{s^2+a}.
$$

**3.** $\frac{d}{d\beta}\log(\beta e^{-\beta})=1/\beta-1=0$, so the mode is $\beta=1$. The density of $u=\log\beta$ is $\pi(e^u)e^u\propto e^{2u-e^u}$, and $\frac{d}{du}(2u-e^u)=0$ gives $e^u=2$, i.e. $\beta=2$. Same posterior, different "MAP". The mean is not invariant under nonlinear maps either. The median is invariant under monotone maps.

So the identification "penalty = negative log prior" holds *in a chosen parametrization*. Weight decay is a Gaussian prior on the weights, which in a network induces some prior on functions that depends on the architecture. The MAP estimate under it is not an invariant summary of the posterior.
"""
    ),
    markdown(
        r"""
### 4.B. As a constraint

#### 4.B.i. The Lagrange multiplier and its monotone map

1. For $c>0$, show that a minimizer of $\hat R_n(b)=\frac1n\|y-Xb\|^2$ subject to $\|b\|^2\le c$ is $\hat\beta_\lambda$, where $\lambda$ is the KKT multiplier. Interpret $\hat\beta_0=X^+y$ when the design is rank deficient.
2. Show that $c(\lambda)=\|\hat\beta_\lambda\|^2$ is strictly decreasing when $X^\top y\neq0$. Hence $\lambda(c)$ is well defined for $c<\|X^+y\|^2$, and $\lambda=0$ for $c\ge\|X^+y\|^2$.
3. In this view, what explains why the constrained estimator generalizes? Which objection from §2 applies to that explanation?

<details><summary>Hint</summary>

For 1, write the Lagrangian $\hat R_n(b)+\lambda(\|b\|^2-c)$ with $\lambda\ge0$, set its gradient to zero, and use complementary slackness. For 2, write $\|\hat\beta_\lambda\|^2$ in the SVD of $X$ and look at each term as a function of $\lambda$. For 3, recall §1.A.i.3 and §2.C; the Rademacher complexity of a norm ball of linear predictors is in UML §26.2.
</details>
"""
    ),
    answer(
        r"""
**1.** The problem is convex, and Slater's condition holds for $c>0$. The Lagrangian is $\hat R_n(b)+\lambda(\|b\|^2-c)$ with $\lambda\ge0$. Stationarity gives $-\frac2nX^\top(y-Xb)+2\lambda b=0$, i.e. $b=(X^\top X+n\lambda I)^{-1}X^\top y=\hat\beta_\lambda$ for $\lambda>0$. For $\lambda=0$, use the minimum-norm least-squares solution $X^+y$; other feasible least-squares solutions may also minimize when $X$ is rank deficient. Complementary slackness says either $\lambda=0$ (the constraint need not be strictly slack) or $\|\hat\beta_\lambda\|^2=c$.

**2.** In the SVD, $\|\hat\beta_\lambda\|^2=\sum_k\frac{s_k^2(u_k^\top y)^2}{(s_k^2+n\lambda)^2}$. Each term is non-increasing in $\lambda$, and at least one is strictly decreasing when $X^\top y\ne0$. As $\lambda\downarrow0$ the norm rises to $\|X^+y\|^2$, the minimum-norm least-squares solution, and as $\lambda\to\infty$ it falls to $0$. So the map from $c$ to $\lambda$ is a decreasing bijection on $(0,\|X^+y\|^2)$. Tighter constraints need bigger multipliers.

**3.** The explanation is capacity. The norm ball $\{x\mapsto x^\top b:\|b\|\le B\}$ has Rademacher complexity of order $B\sqrt{\mathbb E\|x\|^2/n}$ (quoted), which controls prediction complexity. To turn this into a squared-loss gap bound, additionally control the input norms and labels (or suitable tails), so the loss is Lipschitz on the allowed prediction range. Small enough complexity then gives an excess-risk bound via §1.A.i.3. This inherits two problems. The norm depends on the parametrization (4.A.i.3, 4.E), and Zhang et al.'s objection applies: nets with weight decay still fit random labels, so the constraint that is actually active is not small.
"""
    ),
    markdown(
        r"""
### 4.C. As a spectral filter, and early stopping

#### 4.C.i. Two filters

1. Show that $\hat\beta_\lambda=\sum_{k:\,s_k>0}\phi_k\,\dfrac{u_k^\top y}{s_k}\,v_k$ with $\phi_k^{\mathrm{ridge}}=\dfrac{s_k^2}{s_k^2+n\lambda}$. The factor $(u_k^\top y)/s_k$ is the unregularized coefficient.
2. Run gradient flow $\dot\beta=-\nabla L(\beta)$ on $L(\beta)=\frac1{2n}\|X\beta-y\|^2$, starting from $\beta(0)=0$. Show that $\beta(t)$ has the same form with $\phi_k^{\mathrm{GF}}(t)=1-e^{-s_k^2t/n}$. For gradient descent with step $\eta$ and $T$ steps, show $\phi_k=1-(1-\eta s_k^2/n)^T$.
3. Show that the two filters cross $\tfrac12$ at the same $s^2$ if and only if $t=\ln2/\lambda$. With that matching, write both in terms of $x=s^2/(n\lambda)$, and compare them for $x\ll1$ and for $x\gg1$. Which filter passes more of the weak directions? Which shrinks the strong directions more?
4. Let $p>n$. Show that gradient flow from $\beta(0)=0$ converges to the minimum-norm interpolant $X^+y$. What is the limit from a nonzero initialization $\beta_0$?

<details><summary>Hint</summary>

Work in the coordinates $b_k=v_k^\top\beta$, where $X^\top X$ is diagonal and $X^\top y=\sum_ks_k(u_k^\top y)v_k$; gradient flow then decouples into one scalar linear ODE per $k$. For 3, set each filter equal to $\tfrac12$ and solve for $s^2$. For 4, in which subspace does $\dot\beta$ always lie?
</details>

*Further reading:* [Ali, Kolter and Tibshirani (2019)](https://arxiv.org/abs/1810.10082)
compare the risks of gradient flow and ridge along the whole path, under the
calibration $t=1/\lambda$.
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

Gradient flow is the sharper, more step-like filter. The largest disagreements are flow above ridge by about $0.138$ (near $x\approx4.3$) and ridge above flow by about $0.044$ (near $x\approx0.36$). With the naive matching $t=1/\lambda$, flow becomes $1-e^{-x}$ and its half-point moves to $x=\ln2$. "Early stopping is ridge" is right in outline (keep the strong directions, drop the weak ones) and wrong in detail. WB5 §7 derives how the polynomial tail of the ridge filter affects learning-curve exponents.

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
    markdown(
        r"""
### 4.D. Dropout as adaptive $L_2$ (optional)

Drop inputs independently: $\tilde x_{ij}=x_{ij}\xi_{ij}/q$ with
$\xi_{ij}\sim\mathrm{Bernoulli}(q)$, where $q$ is the keep probability. The
$1/q$ keeps $\mathbb E\tilde x_{ij}=x_{ij}$. Part 3 is the main observation of
[Wager, Wang and Liang (2013)](https://arxiv.org/abs/1307.1493).

#### 4.D.i. The penalty hiding in the noise

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
    markdown(
        r"""
### 4.E. Why flatness could matter, and how not to think about it

#### 4.E.i. Local mean excess risk is approximately $\tfrac12\mathrm{Tr}(H\Sigma)$, and sharpness needs a metric

1. Let $\theta^*$ minimize $R$, with $H=\nabla^2R(\theta^*)$, and let the trained parameter be $\hat\theta=\theta^*+\delta$ with $\mathbb E\,\delta\delta^\top=\Sigma$. Show that $\mathbb E R(\hat\theta)-R(\theta^*)\approx\tfrac12\mathrm{Tr}(H\Sigma)$. Check that this is exact for OLS with §1.B's $\Sigma$.
2. Suppose $\hat\theta$ is ERM in a regular model with $\Sigma\approx\frac1nH^{-1}CH^{-1}$ (the sandwich, derived in §7), where $C$ is the covariance of the per-sample gradients at $\theta^*$. Show that $\tfrac12\mathrm{Tr}(H\Sigma)=\frac1{2n}\mathrm{Tr}(H^{-1}C)$, and that if $C=cH$ this equals $cp/(2n)$, whatever the curvature. What has to be true of $\Sigma$ for "flatter is better" to hold?
3. *Dinh et al.'s rescaling.* Take the two-layer scalar linear network $f(x)=w_2w_1x$ with $R(w)=\mathbb E(y-w_2w_1x)^2$ and $a=\mathbb E x^2>0$. Assume $c^*=\mathbb E[xy]/a\ne0$; the minima are the hyperbola $w_1w_2=c^*$. Compute the Hessian at a minimum, show that $\mathrm{Tr}\,H=2a(w_1^2+w_2^2)$ and that one eigenvalue is zero. Under $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ the function does not change. Show that $\mathrm{Tr}\,H$ takes every value in $[4a|c^*|,\infty)$. Why does the same trick work in deep ReLU networks?
4. Let $\theta=g(\varphi)$ be a smooth reparametrization with Jacobian $J$ at the minimum, and let $\delta$ transform as a tangent vector. Show that $\mathrm{Tr}(H\Sigma)$ is invariant while $\mathrm{Tr}\,H$ is not. What does a meaningful flatness claim have to specify?

<details><summary>Hint</summary>

For 1, the gradient vanishes at $\theta^*$, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\,\mathbb E\,\delta\delta^\top)$. For 3, write $R=a(w_1w_2-c^*)^2+\text{const}$. For 4, at a critical point the Hessian transforms as $J^\top HJ$, because the term involving the second derivative of $g$ multiplies $\nabla R=0$.
</details>
"""
    ),
    answer(
        r"""
**1.** $R(\theta^*+\delta)=R(\theta^*)+\nabla R(\theta^*)^\top\delta+\tfrac12\delta^\top H\delta+O(\|\delta\|^3)$. The gradient term vanishes, and $\mathbb E\,\delta^\top H\delta=\mathrm{Tr}(H\Sigma)$. Here $\Sigma$ is the second moment about $\theta^*$, so it includes any bias. For OLS, $R(b)=\sigma^2+\|b-\beta\|^2$ is exactly quadratic with $H=2I$, and $\Sigma=\sigma^2(X^\top X)^{-1}$, so $\tfrac12\mathrm{Tr}(H\Sigma)=\sigma^2\mathrm{Tr}(X^\top X)^{-1}$ exactly, as in §1.B.

**2.** $\tfrac12\mathrm{Tr}\big(H\cdot\tfrac1nH^{-1}CH^{-1}\big)=\tfrac1{2n}\mathrm{Tr}(CH^{-1})$. If $C=cH$, this is $cp/(2n)$. The curvature cancels: a sharp direction is pinned down more tightly by the data ($\Sigma\propto H^{-1}$), and this exactly offsets the larger cost of a given error there. "Flatter is better" needs $\Sigma$ that does not shrink in proportion as $H$ grows. One case is a perturbation scale set by something other than the data's curvature, such as SGD noise at a fixed learning rate, for which §8 finds $\Sigma\propto H^{\gamma-1}$. So "flat minima generalize" is a claim about $H$ *and* $\Sigma$ jointly.

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
### 4.F. Reading

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
            "in 4.E.i.3, generalized to deep rectifier networks. §5 (reparametrizing the model and the "
            "input) is short and makes the same point more generally.",
            tier="core",
        )
        + r"""

#### 4.F.i. Question

Dinh et al. is often cited as "flatness is meaningless". Using 4.E.i.4, say
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
**Where this leaves "the hacky bullshit".** Weight decay, early stopping and
dropout are, in linear models, three ways of choosing a filter on the data's
eigendirections, each in its own metric. The "prior" reading and the "capacity
constraint" reading of the same penalty make different predictions about
random labels (§2.D). Flatness enters through $\Sigma$ and $H$ together.
"""
    ),
    # ------------------------------------------------------------------ §5
    markdown(
        r"""
<a id="wb2-section-5"></a>

## 5. Overfitting and double descent

**The result.** Take least squares with isotropic Gaussian design, in the limit
$p/n\to\gamma$. The ridgeless (minimum-norm) risk is $\sigma^2\gamma/(1-\gamma)$
below the interpolation threshold and $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$
above it. The peak at $\gamma=1$ is all variance, coming from near-zero
eigenvalues of the design. Ridge at its optimal $\lambda^*=\sigma^2\gamma/r^2$
is Bayes-optimal under the isotropic Gaussian prior below, and its risk is monotone in $\gamma$. Past the
peak, when the signal-to-noise ratio exceeds one, the interpolator dips *below*
the null risk $r^2$, with its minimum at $\gamma=r/(r-\sigma)$. So
interpolating noisy data is not automatically harmful. The name "double
descent" is from [Belkin et al. (2019)](https://arxiv.org/abs/1812.11118); the
formulas are those of Hastie et al. (reading in 5.E).

**Conventions (as in Hastie et al.).** Rows $x_i\sim\mathcal N(0,I_p)$, noise
$\varepsilon\sim\mathcal N(0,\sigma^2)$, and $\|\beta\|^2=r^2$, with $r,\sigma>0$. The risk is
$\mathcal R=\mathbb E\big[(x_0^\top\hat\beta-x_0^\top\beta)^2\big]=\mathbb E\|\hat\beta-\beta\|^2$,
which *excludes* the irreducible $\sigma^2$: it is §1's excess risk. Ridge is
$(X^\top X+n\lambda I)^{-1}X^\top y$. Limits are $p,n\to\infty$ with
$p/n\to\gamma$. The null estimator $\hat\beta=0$ has risk $r^2$.

### 5.A. Below the threshold

#### 5.A.i. The left branch

1. From §1.B.ii ($\mathbb E\,\mathcal E_n=\sigma^2p/(n-p-1)$), derive the limit $\sigma^2\gamma/(1-\gamma)$. Explain the divergence in terms of the spectrum of $X^\top X/n$.
2. What is the bias of OLS for $\gamma<1$?
"""
    ),
    answer(
        r"""
**1.** $\sigma^2\frac{p}{n-p-1}=\sigma^2\frac{p/n}{1-p/n-1/n}\to\sigma^2\frac{\gamma}{1-\gamma}$. The empirical spectrum of $X^\top X/n$ converges to the Marchenko–Pastur law on $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$ (quoted). The risk is $\frac{\sigma^2}{n}\sum_i1/\lambda_i(X^\top X/n)$, and as $\gamma\to1$ the lower edge tends to $0$, so the smallest eigenvalues blow up the sum. Consistently, $\frac1p\mathrm{Tr}(X^\top X/n)^{-1}\to\frac1{1-\gamma}$.

**2.** Zero: OLS is unbiased whenever $X$ has full column rank (§1.B.i). Below the threshold the whole risk is variance.
"""
    ),
    markdown(
        r"""
### 5.B. Above the threshold: minimum-norm interpolation

#### 5.B.i. The right branch, bias and variance

For $p>n$, $X$ has full row rank almost surely, and infinitely many $b$ interpolate the data.

1. Show that the minimum-norm least-squares solution is $\hat\beta=X^+y=X^\top(XX^\top)^{-1}y$ and that it interpolates. It is also the $\lambda\to0^+$ limit of ridge, and the limit of gradient descent from $0$ (§4.C.i.4).
2. Let $P=X^+X$ be the orthogonal projection onto the row space of $X$. Show that $\hat\beta-\beta=-(I-P)\beta+X^+\varepsilon$, with the two terms orthogonal. So $\mathcal R=\underbrace{\mathbb E\|(I-P)\beta\|^2}_{\text{conditional bias}^2}+\underbrace{\sigma^2\,\mathbb E\,\mathrm{Tr}(XX^\top)^{-1}}_{\text{variance}}$.
3. *Bias.* Show that $\mathbb E P=(n/p)I$, so that $\text{bias}^2=r^2(1-n/p)\to r^2(1-1/\gamma)$.
4. *Variance.* $XX^\top$ is an $n\times n$ Wishart matrix with $p$ degrees of freedom. Reuse §1.B.ii with the roles of $n$ and $p$ swapped to get $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$.
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

**4.** For $p>n+1$, by §1.B.ii.1 with $(n,p)\to(p,n)$: $\mathbb E(XX^\top)^{-1}=I_n/(p-n-1)$, so the variance is $\sigma^2n/(p-n-1)\to\sigma^2/(\gamma-1)$. (Hastie et al. state this branch as part of their Theorem 1. In the isotropic Gaussian case it is a two-line consequence of the same Wishart fact.)

**5.**

$$
\mathcal R(\gamma)=\begin{cases}\sigma^2\dfrac{\gamma}{1-\gamma},&\gamma<1,\\[2mm] r^2\Big(1-\dfrac1\gamma\Big)+\dfrac{\sigma^2}{\gamma-1},&\gamma>1.\end{cases}
$$

The risk starts at $0$, rises, and diverges at $\gamma=1$ from both sides; the divergence is entirely variance. Past the threshold, the bias grows from $0$ toward $r^2$ while the variance falls from $\infty$ toward $0$. As $\gamma\to\infty$ the risk tends to $r^2$, the null risk: the projection captures a vanishing fraction of $\beta$, and the noise is spread over so many directions that it does no harm. The "second descent" is the variance falling faster than the bias rises.
"""
    ),
    markdown(
        r"""
### 5.C. The second descent and the cure

#### 5.C.i. When does interpolation beat doing nothing, and what removes the peak?

*Source for 3:* [Nakkiran, Venkat, Kakade and Ma (2020)](https://arxiv.org/abs/2003.01897)
prove that optimally tuned ridge is monotone in sample size and model size in
isotropic linear regression.

1. For $\gamma>1$, minimize $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$. Show that if $r>\sigma$ (signal-to-noise ratio $r^2/\sigma^2>1$), the minimum is at $\gamma=r/(r-\sigma)$, with value $2r\sigma-\sigma^2<r^2$. Show that if $r\le\sigma$, the risk decreases monotonically toward $r^2$ from above.
2. Show that $\lambda^*=\sigma^2\gamma/r^2$ is §4.A's $\lambda=\sigma^2/(n\tau^2)$ for the prior $\beta\sim\mathcal N(0,(r^2/p)I)$, which has $\mathbb E\|\beta\|^2=r^2$. Hence ridge at $\lambda^*$ is the posterior mean, the Bayes-optimal estimator under squared loss for that prior.
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

**2.** With $\tau^2=r^2/p$: $\lambda=\frac{\sigma^2}{n\tau^2}=\frac{\sigma^2p}{nr^2}=\frac{\sigma^2\gamma}{r^2}=\lambda^*$. By §4.A.i.1, ridge at $\lambda^*$ is the posterior mean, which minimizes $\mathbb E\|b-\beta\|^2$ over all estimators under that prior. (This is also §3.B.i.3's best common penalty $n\lambda=\sigma^2/\tau^2$.)

**3.** (a) Ridge and min-norm are both rotation-equivariant, and the design is rotation-invariant, so each one's risk at fixed $\beta$ depends only on $\|\beta\|$. Averaged over the prior, ridge($\lambda^*$) is no worse than any estimator. For either of these linear estimators, the noise- and design-averaged risk has the form $A\|\beta\|^2+B$: isotropy makes the squared-bias matrix a scalar multiple of $I$, and the noise variance is independent of $\beta$. Thus averaging over a prior with $\mathbb E\|\beta\|^2=r^2$ gives exactly its risk at any fixed vector of norm $r$, already at finite $p$ (when the risks are finite). The Bayes comparison therefore gives the fixed-radius comparison too. This does not make ridge Bayes-optimal under the uniform-on-a-sphere prior among all estimators.

(b) A Bayes estimator given $n+1$ samples can ignore the last one, so its Bayes risk is at most that of the Bayes estimator with $n$ samples. More data never hurts the posterior mean. The limiting curve depends on $\gamma$ alone, and at fixed $p$ increasing $n$ decreases $\gamma$, so the optimal-ridge risk is non-decreasing in $\gamma$. There is no peak.

The double-descent peak is therefore a property of the minimum-norm estimator, a variance divergence at $p\approx n$ that properly tuned regularization removes, and not a property of the problem.
"""
    ),
    markdown(
        r"""
#### 5.C.ii. The curves at two signal-to-noise ratios

For $r^2/\sigma^2=5$ and for $r^2/\sigma^2=0.5$, use 5.C.i.1 to locate the
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
    markdown(
        r"""
### 5.D. What regime are we in?

Classical asymptotics fix $p$ and send $n\to\infty$. §5 lives at $p/n=O(1)$.
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

#### 5.D.i. Question

Using §3.A, describe a measurement that would tell you whether a network's
double-descent peak is a variance divergence of the kind derived here.
"""
    ),
    answer(
        r"""
Train several networks at the same size on independent data draws (or with different seeds and different label noise) near the apparent peak. Decompose the test loss into the loss of the ensemble average (bias plus noise) and the spread across the ensemble (variance), as in §3.A. If the peak is a variance divergence, the spread peaks there while the ensemble-average loss does not. Weight decay tuned per size should then flatten the peak. A second check is the spectrum of the relevant kernel or feature covariance at the peak: there should be near-zero eigenvalues.
"""
    ),
    markdown(
        r"""
### 5.E. Reading (optional)

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
            "derived in 5.B, for general i.i.d. entries rather than only Gaussian ones). §5 on misspecified "
            "models shows when the global minimum can sit at γ > 1. §4 (general covariance) and §8 "
            "(nonlinear random features) are for those heading to WB5.",
            tier="optional",
        )
    ),
]
