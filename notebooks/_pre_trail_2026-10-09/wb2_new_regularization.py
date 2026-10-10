"""WB2 new material: regularization (1.D opener, 1.D.3 L1 and sparsity, 1.D.5 input noise and augmentation).

Formulas are checked numerically by checks/wb2_new_regularization_check.py.
"""
from nbkit import *  # noqa: F401,F403

GBC7 = "https://www.deeplearningbook.org/contents/regularization.html"
GBC8 = "https://www.deeplearningbook.org/contents/optimization.html"

OPENER = [
    markdown(
        r"""
<a id="wb2-section-1d"></a>

### 1.D. Regularization

Every network that is trained in practice is regularized, usually by several
things at once: weight decay, data augmentation, dropout, early stopping,
label smoothing, and choices nobody designed as regularizers, such as the
optimizer and the architecture. This subsection asks what each of these does
to the learned function, and which of them are the same thing in different
clothes. It first surveys the techniques, with pointers to the standard
textbook chapter, and then derives the main mechanisms exactly in linear
regression. There each technique turns into a precise statement about which
directions of the data get shrunk, and by how much.

**What counts as regularization.** Goodfellow, Bengio and Courville (*Deep
Learning*, 2016, §5.2.2) define it as "any modification we make to a learning
algorithm that is intended to reduce its generalization error but not its
training error". Their chapter 7 opens with the same idea: strategies
"designed to reduce the test error, possibly at the expense of increased
training error". The definition is about purpose, not form. Penalties on the
weights count, but so do changes to the data, the optimizer, the
architecture, and averaging several models.

**Why it is needed.** There are three reasons, from mild to severe.

- *Variance.* An estimate fitted to $n$ noisy examples is itself noisy, and the noise is worst in the directions the data barely pin down. For least squares at a fixed design, the parameter error is $\mathbb E\|\hat\beta_{\mathrm{OLS}}-\beta\|^2=\sigma^2\mathrm{Tr}(X^\top X)^{-1}=\sigma^2\sum_k s_k^{-2}$, where the $s_k$ are the singular values of the design (defined below), so the weakest directions dominate. Shrinking those directions adds a little bias and removes a lot of variance. For ridge, a small enough penalty lowers the error for *every* true $\beta$ ([Hoerl and Kennard, 1970](https://doi.org/10.1080/00401706.1970.10488634)).
- *Ill-posedness.* If $p>n$, or more generally if $X$ is rank deficient, the training loss has a whole affine subspace of minimizers. The data cannot choose among them, so something else has to. This was Tikhonov's motivation for ill-posed inverse problems, which is why ridge is also called *Tikhonov regularization*.
- *Deep learning lives at $p\gtrsim n$.* Networks routinely have more parameters than training examples and can fit random labels. Which of the many zero-training-loss solutions you get is decided entirely by the regularizers, explicit and implicit.

**The toolbox.** The table lists the techniques in common use, with one line
on how each works. "GBC" is Goodfellow, Bengio and Courville's chapter 7,
[*Regularization for Deep Learning*]("""
        + GBC7
        + r"""), free online; the section links jump to the right page. The last
column says where this subsection derives the mechanism in a linear model.

| Technique | What it does | GBC | Here |
|---|---|---|---|
| $L^2$ penalty ("weight decay") | Add $\lambda\lVert\theta\rVert_2^2$ to the loss. Shrinks every weight toward 0, most strongly in directions the data constrain weakly. | [§7.1.1]("""
        + GBC7
        + r"""#pf4) | 1.D.1, 1.D.2, 1.D.4 |
| $L^1$ penalty | Add $\lambda\lVert\theta\rVert_1$. Sets many weights exactly to 0, so it selects features as well as shrinking them. | [§7.1.2]("""
        + GBC7
        + r"""#pf7) | 1.D.3 |
| Norm constraints | Keep $\lVert w\rVert\le c$ (often per unit, "max-norm") by projecting after each step: the hard form of a penalty. | [§7.2]("""
        + GBC7
        + r"""#pfa) | 1.D.2 |
| Choosing among many minimizers | When the problem is underdetermined, pick one solution, e.g. the minimum-norm one $X^+y$, which is the $\lambda\to0$ limit of ridge. | [§7.3]("""
        + GBC7
        + r"""#pfc) | 1.D.4 |
| Data augmentation | Train on transformed copies (flips, crops, shifts) whose label should not change. Builds in an invariance. | [§7.4]("""
        + GBC7
        + r"""#pfd) | 1.D.5 |
| Noise injection | Add noise to inputs, hidden units or weights during training. For a linear model, input noise is exactly an $L^2$ penalty; for a nonlinear one, it acts like a smoothness penalty only approximately. | [§7.5]("""
        + GBC7
        + r"""#pff) | 1.D.5 |
| Label smoothing | With $K$ classes, train toward soft targets $\epsilon/(K-1)$ and $1-\epsilon$ instead of 0 and 1, so the loss stops rewarding ever-larger logits. | [§7.5.1]("""
        + GBC7
        + r"""#pf10) | |
| Semi-supervised and multitask learning | Share a representation with unlabelled data or other tasks, whose data then constrain the shared parameters. | [§7.6–7.7]("""
        + GBC7
        + r"""#pf11) | |
| Early stopping | Stop when validation loss stops improving. For a quadratic loss, roughly equivalent to $L^2$. | [§7.8]("""
        + GBC7
        + r"""#pf12) | 1.D.4 |
| Parameter tying and sharing | Force groups of weights to be equal (a convolution uses the same filter at every position), which builds in a symmetry. | [§7.9]("""
        + GBC7
        + r"""#pf1a) | |
| Sparse representations | Penalize the activations rather than the weights, e.g. $L^1$ on hidden units. | [§7.10]("""
        + GBC7
        + r"""#pf1c) | |
| Bagging and ensembles | Train several models (on resampled data, or from different seeds) and average them. Averaging cuts variance. | [§7.11]("""
        + GBC7
        + r"""#pf1e) | |
| Dropout | Zero random units during training. A cheap approximation to bagging many subnetworks; in a linear model, $L^2$ in a data-dependent metric. | [§7.12]("""
        + GBC7
        + r"""#pf20) | 1.D.6 |
| Adversarial training | Train on worst-case perturbations of each input within a small ball, which makes $f$ locally flat in input space. | [§7.13]("""
        + GBC7
        + r"""#pf2a) | |
| Tangent propagation | Penalize the derivative of $f$ along directions of known invariance: the infinitesimal version of augmentation. | [§7.14]("""
        + GBC7
        + r"""#pf2c) | |
| Decoupled weight decay (AdamW) | With plain SGD, an $L^2$ penalty and shrinking the weights by a fixed factor each step are the same. With Adam they are not, because Adam divides the penalty's gradient by a per-coordinate scale. AdamW applies the decay directly. | [Loshchilov and Hutter (2019)](https://arxiv.org/abs/1711.05101) | |
| Implicit regularizers | Not designed as regularizers, but they act like them: which minimizer gradient descent finds, minibatch noise ([GBC §8.1.3]("""
        + GBC8
        + r"""#pf4)), batch normalization ([GBC §8.7.1]("""
        + GBC8
        + r"""#pf2b); [Ioffe and Szegedy (2015)](https://arxiv.org/abs/1502.03167) report it can replace dropout), and the architecture itself. | | 1.D.4 |

**What this subsection does.** It works in linear regression, where every
claim can be checked exactly.

- **1.D.1** Ridge as a Gaussian prior: the MAP estimate and the posterior, and why "penalty = log prior" depends on the parametrization.
- **1.D.2** Ridge as a norm constraint, with the Lagrange multiplier as the exchange rate.
- **1.D.3** $L^1$ and sparsity: soft thresholding, and why exact zeros appear.
- **1.D.4** Ridge as a spectral filter on the data's singular directions; early stopping as a similar, but not identical, filter; gradient descent's choice of minimizer.
- **1.D.5** Input noise and augmentation: for a linear model input noise is exactly ridge; for a flexible model the input-gradient penalty matches the noisy objective to leading order but not its minimizer; augmentation by a symmetry penalizes the non-invariant part of the weights.
- **1.D.6** Dropout (optional): in a linear model it is ridge in a data-dependent metric.
- **1.D.7** Flatness: when the sharpness of a minimum can matter for generalization, and why any sharpness claim needs a metric.
- **1.D.8** A reading: Dinh et al., "Sharp minima can generalize".

**Setup used throughout 1.D.** There are $n$ training examples with inputs
$x_i\in\mathbb R^p$ and targets $y_i\in\mathbb R$. Stack the inputs as the rows
of the design matrix $X\in\mathbb R^{n\times p}$ and the targets as
$y\in\mathbb R^n$. The data follow the linear model $y=X\beta+\varepsilon$, where
$\beta\in\mathbb R^p$ is the true coefficient vector and the noise
$\varepsilon$ has mean $0$ and covariance $\sigma^2I_n$. The design is treated as
fixed unless a part says otherwise. Ridge regression with penalty $\lambda>0$
is

$$
\hat\beta_\lambda=\arg\min_b\ \frac1n\|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+n\lambda I)^{-1}X^\top y ,
$$

and $\lambda=0$ (with $X$ of full column rank) is ordinary least squares
(OLS). Write the thin singular value decomposition as
$X=U\,\mathrm{diag}(s_k)\,V^\top$, with singular values $s_k\ge0$, left singular
vectors $u_k\in\mathbb R^n$ (columns of $U$) and right singular vectors
$v_k\in\mathbb R^p$ (columns of $V$). Then $X^\top X=V\,\mathrm{diag}(s_k^2)\,V^\top$,
and a direction $v_k$ with small $s_k$ is one the data barely constrain. When
$x\sim\mathcal N(0,I_p)$, the parameter error $\mathbb E\|\hat\beta-\beta\|^2$
equals the excess test risk, which is why the exercises track it.
"""
        + references(
            book(
                "Deep Learning",
                GBC7,
                "Goodfellow, Bengio and Courville (2016; free online)",
                "ch. 7, intro and §7.1, §7.8, §7.12",
            ),
            book(
                "The Elements of Statistical Learning",
                "https://hastie.su.domains/ElemStatLearn/",
                "Hastie, Tibshirani and Friedman (2nd ed. 2009; free PDF)",
                "§3.4.1–3.4.3",
            ),
        )
    ),
]

L1 = [
    markdown(
        r"""
#### 1.D.3. L1 and sparsity

Replace the squared norm in the ridge penalty by the sum of absolute values,
and the estimator starts setting coefficients exactly to zero: it selects
features and shrinks them at once. This is the **lasso**
([Tibshirani, 1996](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)). This
group shows why one penalty gives exact zeros and the other never does, and
what the usual Bayesian reading of the lasso is worth.

Setup: the linear model $y=X\beta+\varepsilon$ with $X\in\mathbb R^{n\times p}$,
noise variance $\sigma^2$, and ridge
$\hat\beta_\lambda=\arg\min_b\frac1n\|y-Xb\|^2+\lambda\|b\|^2$, as in the 1.D
setup. The lasso is

$$
\hat\beta^{\mathrm{lasso}}=\arg\min_b\ \frac1{2n}\|y-Xb\|^2+\lambda\|b\|_1,\qquad \|b\|_1=\sum_j|b_j| .
$$

The factor $\tfrac12$ is a convention that makes the threshold below come out
as $\lambda$. In general the lasso has no closed form. Parts 1, 2 and 4 use an
*orthonormal design*, $X^\top X=nI_p$ (which needs $n\ge p$), where it does.
Then the OLS estimate is $z=X^\top y/n$.

1. With $X^\top X=nI$, show that each estimator acts on each coordinate $z_j$ separately:
   - ridge gives $z_j/(1+\lambda)$;
   - the lasso gives the *soft threshold* $\mathrm{sign}(z_j)\,(|z_j|-\lambda)_+$, where $(t)_+=\max(t,0)$;
   - best-subset selection with an $L^0$ penalty, $\arg\min_b\frac1{2n}\|y-Xb\|^2+\frac{\lambda^2}2\|b\|_0$ with $\|b\|_0$ the number of nonzero entries, gives the *hard threshold* $z_j\,\mathbf 1[|z_j|>\lambda]$.

   Sketch the three as functions of $z_j$. Which of them is biased for large coefficients, and by how much?
2. Give the coefficients independent Laplace priors with density $\frac1{2\tau}e^{-|\beta_j|/\tau}$ (scale $\tau>0$), and take the likelihood $y\mid\beta\sim\mathcal N(X\beta,\sigma^2I)$, for general $X$. Show that the MAP estimate is the lasso with $\lambda=\sigma^2/(n\tau)$.
3. For general $X$, show that $\hat b$ minimizes the lasso objective if and only if, for every $j$,
$$
\tfrac1n\,x_{\cdot j}^\top(y-X\hat b)=\lambda\,\mathrm{sign}(\hat b_j)\ \text{ if }\hat b_j\ne0,\qquad \big|\tfrac1n\,x_{\cdot j}^\top(y-X\hat b)\big|\le\lambda\ \text{ if }\hat b_j=0,
$$
where $x_{\cdot j}$ is the $j$-th column of $X$. In words: a feature stays out until its correlation with the residual reaches $\lambda$. Use this to explain why the lasso gives $\hat b_j=0$ for a whole open set of data $y$, while ridge with $\lambda>0$ gives $\hat\beta_{\lambda,j}=0$ only on a set of probability zero. Draw the constrained version (minimize the loss subject to $\|b\|_1\le c$) in two dimensions and say where the zeros come from.
4. *A caveat about the Bayesian reading.* In the orthonormal case, show that the posterior of part 2 factorizes over coordinates, with $\pi(\beta_j\mid y)\propto\exp\!\big(-\frac{n}{2\sigma^2}(\beta_j-z_j)^2-|\beta_j|/\tau\big)$. Show that the posterior mean of $\beta_j$ has the sign of $z_j$ and is never exactly zero unless $z_j=0$. Is a draw from the Laplace prior ever sparse? What does this say about "the lasso is Bayesian inference under a sparsity prior"?

<details><summary>Hint</summary>

For 1, expand $\frac1{2n}\|y-Xb\|^2=\frac1{2n}\|y\|^2-b^\top z+\frac12\|b\|^2$, which is $\frac12\|b-z\|^2$ plus a constant. For the scalar lasso problem $\frac12(b-z)^2+\lambda|b|$, try $b>0$ and $b<0$ separately, and otherwise $b=0$. For 3, the objective is convex, so $\hat b$ is a minimizer iff $0$ lies in its subdifferential; the subdifferential of $|b|$ is $\{\mathrm{sign}(b)\}$ for $b\ne0$ and the interval $[-1,1]$ at $b=0$. For 4, compare the posterior density at $\beta$ and at $-\beta$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $X^\top X=nI$ and $z=X^\top y/n$, $\frac1{2n}\|y-Xb\|^2=\frac12\|b-z\|^2+\text{const}$, and every penalty here is a sum over coordinates, so each problem splits into scalar problems in $(b_j,z_j)$. Drop the index $j$.

- *Ridge.* $\frac1n\|y-Xb\|^2+\lambda\|b\|^2=\|b-z\|^2+\lambda\|b\|^2+\text{const}$, minimized at $b=z/(1+\lambda)$. This is also $(X^\top X+n\lambda I)^{-1}X^\top y=(nI+n\lambda I)^{-1}nz$.
- *Lasso.* Minimize $h(b)=\frac12(b-z)^2+\lambda|b|$. On $b>0$, $h'(b)=b-z+\lambda$ vanishes at $b=z-\lambda$, which lies in $b>0$ iff $z>\lambda$. On $b<0$ the stationary point is $b=z+\lambda$, valid iff $z<-\lambda$. If $|z|\le\lambda$, then $h'<0$ on $b<0$ and $h'>0$ on $b>0$, so the minimum is at the kink, $b=0$. Together, $b=\mathrm{sign}(z)(|z|-\lambda)_+$.
- *$L^0$.* A coordinate set to $0$ costs $\frac12z^2$. A nonzero coordinate pays the penalty $\frac{\lambda^2}2$ whatever its size, so it might as well be $b=z$, costing $\frac{\lambda^2}2$. Keep it iff $z^2>\lambda^2$: $b=z\,\mathbf 1[|z|>\lambda]$. (Best subset of a fixed size $M$ likewise keeps the $M$ largest $|z_j|$.)

The sketches: ridge is a line through the origin with slope $1/(1+\lambda)$. Soft thresholding is $0$ on $[-\lambda,\lambda]$ and then runs parallel to the diagonal, shifted toward zero by $\lambda$. Hard thresholding is $0$ on $[-\lambda,\lambda]$ and then jumps onto the diagonal. For large $|z|$, ridge pulls the estimate toward zero by $\frac{\lambda}{1+\lambda}|z|$, which grows in proportion to the coefficient; the lasso pulls it in by the constant $\lambda$; hard thresholding is unbiased for large coefficients but discontinuous at the threshold, so a small change in the data can switch a coefficient between $0$ and $\pm\lambda$. The lasso sits between the other two: continuous like ridge, sparse like subset selection. This is ESL's Table 3.4.

**2.** The negative log posterior is
$$
\frac1{2\sigma^2}\|y-X\beta\|^2+\frac1\tau\|\beta\|_1+\text{const}.
$$
Multiplying by $\sigma^2/n$ gives $\frac1{2n}\|y-X\beta\|^2+\frac{\sigma^2}{n\tau}\|\beta\|_1$, the lasso objective with $\lambda=\sigma^2/(n\tau)$. As with ridge and the Gaussian prior (1.D.1), "penalty = negative log prior" holds in the parametrization you chose.

**3.** The objective is convex, so $\hat b$ minimizes it iff $0\in-\frac1nX^\top(y-X\hat b)+\lambda\,\partial\|\hat b\|_1$. The subdifferential of $\|b\|_1$ is the product over $j$ of $\{\mathrm{sign}(b_j)\}$ when $b_j\ne0$ and $[-1,1]$ when $b_j=0$. Reading off coordinate $j$ gives exactly the two stated conditions.

*Why zeros are robust for the lasso.* The condition for $\hat b_j=0$ is an inequality, which typically holds strictly. In the orthonormal case it reads $|z_j|\le\lambda$: a whole interval of $z_j$, which has positive probability under any continuous noise. For general $X$, the residual $y-X\hat b$ is the same for every lasso solution and depends continuously on $y$, and a coordinate whose inequality is strict must be zero in every solution (a nonzero $\hat b_j$ would need equality). So if the inequality holds strictly at some $y$, it still holds after a small change of $y$, and the same coordinates stay at zero on an open set of $y$. Ridge is different. $\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$ is linear in $y$, so $\hat\beta_{\lambda,j}=0$ is one linear equation in $y$: a hyperplane, of probability zero under continuous noise (unless that row of the matrix vanishes). In the orthonormal case, $z_j/(1+\lambda)=0$ only at $z_j=0$.

*The picture.* In the constrained form, the level sets of $\|y-Xb\|^2$ are ellipses centred at the OLS solution, and the solution is where the smallest ellipse touches the ball $\{\|b\|_1\le c\}$. In two dimensions that ball is a diamond with corners on the axes. An ellipse growing from a generic centre often first touches a corner, where one coordinate is exactly $0$; it does so for a whole region of centres. The $L^2$ ball is round, so the touching point lies on an axis only if the centre is specially placed. In $p$ dimensions the $L^1$ ball has corners and low-dimensional faces on every coordinate subspace, which is where sparse solutions live. This is ESL's Figure 3.11.

**4.** With $X^\top X=nI$, $\|y-X\beta\|^2=n\|\beta-z\|^2+\text{const}$, so the posterior is a product of the stated one-dimensional densities. Take $z_j>0$ and drop the index. For $\beta>0$,
$$
\frac{\pi(\beta\mid y)}{\pi(-\beta\mid y)}=\exp\!\Big(-\frac{n}{2\sigma^2}\big[(\beta-z)^2-(\beta+z)^2\big]\Big)=\exp\!\Big(\frac{2n\beta z}{\sigma^2}\Big)>1 ,
$$
so $\mathbb E[\beta\mid y]=\int_0^\infty\beta\,\big[\pi(\beta\mid y)-\pi(-\beta\mid y)\big]\,d\beta>0$. By symmetry the mean is negative for $z<0$, and it is zero only at $z=0$. The same comparison gives $\Pr(\beta>0\mid y)>\tfrac12$, so the posterior median is not zero either. The posterior mean is the Bayes estimate under squared loss, and it is never sparse; only the mode is.

A draw from the Laplace prior is nonzero with probability one, so the prior does not believe in sparsity at all. Exact zeros come from taking the mode of a density with a kink at $0$. The lasso is better understood as a convex surrogate for best-subset selection (part 1) than as Bayesian inference. If you really believe many coefficients are exactly zero, the matching prior puts a point mass at zero (a "spike-and-slab" prior). Its posterior probability that $\beta_j=0$ is then meaningful, but its posterior mean is still not sparse.
"""
    ),
]

NOISE = [
    markdown(
        r"""
#### 1.D.5. Input noise and data augmentation as penalties

Two common regularizers change the data rather than the loss: adding noise to
the inputs, and training on transformed copies of each example (mirror images,
crops, shifts) whose labels should not change. In a linear model each one
becomes an explicit penalty you can write down. Gaussian input noise is exactly
ridge. Augmentation by a symmetry penalizes the part of the weights that is not
invariant, and when every transformation is used equally often that penalty
becomes a hard constraint. For a nonlinear model,
[Bishop (1995)](https://doi.org/10.1162/neco.1995.7.1.108) argued that small
input noise is equivalent to adding a penalty on the input gradient,
$s^2\,\mathbb E\|\nabla_xf\|^2$ (a *Tikhonov* penalty, i.e. a penalty on
derivatives of $f$). Parts 2 and 3 check this: the two objectives agree to
order $s^2$ near the regression function, but their minimizers differ at that
same order.

Setup: training pairs $(x_i,y_i)$, $i=1,\dots,n$, with $x_i\in\mathbb R^p$,
squared loss, and $X\in\mathbb R^{n\times p}$ the matrix with rows $x_i^\top$.
A linear predictor is $f(x)=\beta^\top x$. *Input noise* replaces $x_i$ by
$\tilde x_i=x_i+\xi_i$, where the $\xi_i$ are i.i.d. with mean $0$ and covariance
$s^2I_p$, independent of the data. We study the training loss averaged over
the noise, which is what drawing fresh noise at every step approximates over
many epochs.

1. Show that, exactly,
$$
\mathbb E_\xi\,\frac1n\sum_i(y_i-\tilde x_i^\top\beta)^2=\frac1n\|y-X\beta\|^2+s^2\|\beta\|^2 ,
$$
so the minimizer is ridge with $\lambda=s^2$. What penalty does noise with covariance $\Sigma_\xi$ give? (The dropout exercise, 1.D.6, multiplies the inputs by noise instead, and gets a penalty in a data-dependent metric.)
2. Now let $f$ be a smooth nonlinear function of $x$ (its parameters are suppressed), and let $\xi\sim\mathcal N(0,s^2I_p)$. For one example, write $r=y-f(x)$ for the residual, $g=\nabla_xf(x)$ for the input gradient and $\Delta f=\mathrm{Tr}\,\nabla_x^2f(x)$ for the Laplacian. Show that
$$
\mathbb E_\xi\big(y-f(x+\xi)\big)^2=r^2+s^2\big(\|g\|^2-r\,\Delta f\big)+O(s^4).
$$
Check that part 1 is the linear case. Which of the two $s^2$ terms is a penalty, and which is not?
3. *Bishop's argument, and where it stops.* Assume enough data that sums over examples can be replaced by expectations over $(x,y)$, and let $m(x)=\mathbb E[y\mid x]$ be the regression function.
   - (a) Show that the averaged second term is $-s^2\,\mathbb E_x\big[(m(x)-f(x))\,\Delta f(x)\big]$. It vanishes at $f=m$ and is $O(s^4)$ when $f-m=O(s^2)$. Bishop concluded that this term can be dropped: to order $s^2$, the noisy objective and the *Tikhonov objective* $\mathbb E(y-f(x))^2+s^2\,\mathbb E\|\nabla_xf\|^2$ are minimized by the same function.
   - (b) Test this where everything is solvable. Let $x$ be uniform on the circle $[0,2\pi)$, with the noise added modulo $2\pi$, let $y=m(x)+\varepsilon$, and minimize over all smooth periodic $f$. Show that the noisy objective is minimized by $f(t)=\mathbb E[m(x)\mid x+\xi=t]$, which here is $m$ convolved with a Gaussian of variance $s^2$: on the Fourier mode $e^{ikx}$ it multiplies by $e^{-s^2k^2/2}$. Show that the Tikhonov objective $\mathbb E(y-f(x))^2+s^2\,\mathbb E f'(x)^2$ is minimized by the multiplier $1/(1+s^2k^2)$. Expand both to order $s^2$. Do they agree? What did the argument in (a) miss?
4. *Augmentation by a mirror symmetry.* Let $S$ be an orthogonal involution, $S^\top=S$ and $S^2=I$; for example, reversing the order of the pixels in each row of an image (a left–right mirror). Let $P=\frac12(I+S)$ and $Q=\frac12(I-S)$. Augmentation trains on both $(x_i,y_i)$ and $(Sx_i,y_i)$. Show that
$$
\frac1{2n}\sum_i\Big[(y_i-\beta^\top x_i)^2+(y_i-\beta^\top Sx_i)^2\Big]=\frac1n\|y-XP\beta\|^2+\frac1n\|XQ\beta\|^2 .
$$
Deduce that if $XQ$ is injective on the range of $Q$ (enough data), the minimizer has $Q\hat\beta=0$, so the learned predictor is exactly mirror-invariant, and $\hat\beta$ is the least-squares fit to the symmetrized inputs $Px_i$. What does this cost if the true $\beta$ has $Q\beta\ne0$?
5. *Partial augmentation.* Now mirror each example with probability $\alpha\in[0,\tfrac12)$ and keep it otherwise. Work with the population risk, under an input distribution that is mirror-symmetric ($Sx$ has the same distribution as $x$); the labels need not be. Let $\Sigma=\mathbb E[xx^\top]$, assumed invertible. Show that
   - (a) $P\Sigma Q=0$;
   - (b) the minimizer of the augmented risk has the same $P\beta$ as the minimizer of the clean risk, and $Q\beta$ multiplied by $1-2\alpha$;
   - (c) this is also the minimizer of the clean risk plus $\kappa\,\beta^\top Q\Sigma Q\beta$ with $\kappa=2\alpha/(1-2\alpha)$: ridge on the non-invariant part only, in the data's own metric. What happens as $\alpha\to\tfrac12$?

<details><summary>Hint</summary>

For 1, write $\tilde x_i^\top\beta=x_i^\top\beta+\xi_i^\top\beta$ and use $\mathbb E(a-Z)^2=a^2+\mathrm{Var}\,Z$ for a mean-zero $Z$. For 2, Taylor expand $f(x+\xi)$ to third order; odd moments of $\xi$ vanish, and $\mathbb E\,\xi^\top A\xi=s^2\mathrm{Tr}A$. For 3(b), substitute $t=x+\xi$: $t$ is again uniform and independent of $\xi$, and $x=t-\xi$. For the Tikhonov objective, expand $f$ and $m$ in Fourier series and use Parseval; the problem separates by frequency. For 4 and 5, write $\beta=\beta_P+\beta_Q$ with $\beta_P=P\beta$, $\beta_Q=Q\beta$. Then $\beta^\top x=\beta_P^\top x+\beta_Q^\top x$ and $\beta^\top Sx=\beta_P^\top x-\beta_Q^\top x$, because $S\beta_P=\beta_P$ and $S\beta_Q=-\beta_Q$.
</details>
"""
    ),
    answer(
        r"""
**1.** For one example, $y_i-\tilde x_i^\top\beta=(y_i-x_i^\top\beta)-\xi_i^\top\beta$, and $\xi_i^\top\beta$ has mean $0$ and variance $\beta^\top(s^2I)\beta=s^2\|\beta\|^2$. So $\mathbb E_\xi(y_i-\tilde x_i^\top\beta)^2=(y_i-x_i^\top\beta)^2+s^2\|\beta\|^2$, with no approximation, and only the first two moments of the noise matter. Averaging over $i$ gives the claim, which is the ridge objective with $\lambda=s^2$. Its minimizer is $(X^\top X+ns^2I)^{-1}X^\top y$. With noise covariance $\Sigma_\xi$ the penalty is $\beta^\top\Sigma_\xi\beta$, a "generalized ridge": noise along a direction penalizes weight along that direction.

**2.** Taylor expand: $f(x+\xi)=f+g^\top\xi+\frac12\xi^\top H\xi+\frac16T[\xi,\xi,\xi]+O(|\xi|^4)$, with $H=\nabla_x^2f$ and $T$ the third-derivative tensor. Then
$$
\big(y-f(x+\xi)\big)^2=r^2-2r\,g^\top\xi-r\,\xi^\top H\xi-\tfrac13r\,T[\xi,\xi,\xi]+(g^\top\xi)^2+(g^\top\xi)(\xi^\top H\xi)+O(|\xi|^4).
$$
Take expectations. Terms with an odd number of $\xi$'s vanish. $\mathbb E(g^\top\xi)^2=s^2\|g\|^2$ and $\mathbb E\,\xi^\top H\xi=s^2\mathrm{Tr}H=s^2\Delta f$. The remainder has expectation $O(s^4)$. This gives the stated formula. For linear $f$, $g=\beta$ and $H=0$, and the expansion is exact, recovering part 1.

The term $s^2\|g\|^2$ is a penalty: nonnegative, and it rewards functions that change slowly with the input. The term $-s^2r\,\Delta f$ has no sign. It can be lowered by curving $f$ so that $\Delta f$ has the same sign as the residual, so on its own it is not a regularizer.

**3.** (a) $\mathbb E_{x,y}[r\,\Delta f]=\mathbb E_x\big[\mathbb E[y-f(x)\mid x]\,\Delta f(x)\big]=\mathbb E_x\big[(m-f)\Delta f\big]$. This is zero at $f=m$, and if $f-m=O(s^2)$ it is $O(s^2)$, so the term $s^2\,\mathbb E[r\Delta f]$ is $O(s^4)$.

(b) *Noise.* The noisy objective is $\mathbb E_{x,\varepsilon,\xi}\big(m(x)+\varepsilon-f(x+\xi)\big)^2=\sigma^2+\mathbb E_{t,\xi}\big(m(t-\xi)-f(t)\big)^2$, writing $\sigma^2$ for the variance of $\varepsilon$. Here $t=x+\xi$ is uniform on the circle and independent of $\xi$, since $x$ is uniform and independent of $\xi$. For each $t$ the minimizing $f(t)$ is the mean, $f(t)=\mathbb E_\xi\,m(t-\xi)=\mathbb E[m(x)\mid x+\xi=t]$, which is $m$ convolved with the wrapped Gaussian. The Fourier coefficients of a Gaussian of variance $s^2$ are $e^{-s^2k^2/2}$, so $\hat f_k=e^{-s^2k^2/2}\hat m_k$.

*Tikhonov.* By Parseval, with coefficients normalized so that $\mathbb E|f(x)|^2=\sum_k|\hat f_k|^2$, the objective is $\sigma^2+\sum_k\big(|\hat m_k-\hat f_k|^2+s^2k^2|\hat f_k|^2\big)$. Each term is minimized at $\hat f_k=\hat m_k/(1+s^2k^2)$.

*Comparison.* To order $s^2$, $e^{-s^2k^2/2}\approx1-\frac12s^2k^2$ and $1/(1+s^2k^2)\approx1-s^2k^2$. Since multiplying $\hat m_k$ by $-k^2$ is taking a second derivative,
$$
f_{\text{noise}}=m+\tfrac{s^2}2\,m''+O(s^4),\qquad f_{\text{Tikhonov}}=m+s^2\,m''+O(s^4).
$$
They differ at order $s^2$, the same order as the regularization effect itself: the Tikhonov penalty with coefficient $s^2$ smooths twice as much as the noise does.

*What (a) missed.* The dropped term is $O(s^4)$ in *value* near $m$, but its *gradient* with respect to $f$ is $O(s^2)$, the same order as the gradient of $s^2\,\mathbb E f'^2$, so dropping it moves the minimizer at leading order. Keeping it fixes this. On the circle, integrating by parts gives $-\mathbb E[(m-f)f'']=\mathbb E[(m'-f')f']$, so the full second-order objective is
$$
\sigma^2+\mathbb E(m-f)^2+s^2\,\mathbb E[f'^2]+s^2\,\mathbb E[(m'-f')f']=\sigma^2+\mathbb E(m-f)^2+s^2\,\mathbb E[m'f'] ,
$$
whose minimizer satisfies $-2(m-f)-s^2m''=0$, i.e. $f=m+\frac{s^2}2m''$, in agreement with the exact answer. The $s^2$ part of this objective is linear in $f$, not a penalty at all.

So the precise statements are these.

- *Linear model:* input noise with covariance $s^2I$ is ridge with $\lambda=s^2$, exactly (part 1).
- *Flexible model, plenty of data:* noise training converges to the smoothed regression function $\mathbb E[m(x)\mid x+\xi]$, the "target smoothing" picture of [Reed, Marks and Oh (1995)](https://doi.org/10.1109/72.377960).
- *What Bishop's argument does establish:* for every $f$ within $O(s^2)$ of $m$, the noisy objective and the Tikhonov objective differ by $O(s^4)$; and both minimizers are $m+O(s^2)$.
- *What it does not:* the $O(s^2)$ correction to $m$, which is the whole regularization effect, differs. On the uniform circle the Tikhonov minimizer moves twice as far, so there a penalty with coefficient $s^2/2$ instead of $s^2$ would match to order $s^2$. With a non-uniform input density $\rho(x)$ even that fails: to order $s^2$, $\mathbb E[m(x)\mid x+\xi]=m+s^2\big(\tfrac12\Delta m+\nabla\log\rho\cdot\nabla m\big)$, whereas the Tikhonov minimizer with coefficient $c$ is $m+c\,\big(\Delta m+\nabla\log\rho\cdot\nabla m\big)$ to first order in $c$, and no single $c$ matches both terms. The perturbative weight correction in Bishop's §3 is computed from the reduced penalty, so it inherits the same discrepancy.

**4.** Write $\beta_P=P\beta$, $\beta_Q=Q\beta$. $P$ and $Q$ are symmetric, $P^2=P$, $Q^2=Q$, $PQ=0$ and $P+Q=I$ (all from $S^2=I$), so they are complementary orthogonal projections onto the symmetric vectors ($Sv=v$) and the antisymmetric vectors ($Sv=-v$). Since $S\beta_P=\beta_P$ and $S\beta_Q=-\beta_Q$, $\beta^\top Sx=(S\beta)^\top x=\beta_P^\top x-\beta_Q^\top x$. With $a_i=y_i-\beta_P^\top x_i$ and $c_i=\beta_Q^\top x_i$, the two losses are $(a_i-c_i)^2$ and $(a_i+c_i)^2$, whose average is $a_i^2+c_i^2$. Since $\beta_P^\top x_i=(XP\beta)_i$ and $c_i=(XQ\beta)_i$, this is the claim.

The fit term depends only on $P\beta$ and the penalty only on $Q\beta$, and the penalty is minimized, at $0$, exactly when $XQ\beta=0$. If $XQ$ is injective on the range of $Q$, that forces $Q\hat\beta=0$. Then $f(Sx)=\hat\beta^\top Sx=\hat\beta^\top x=f(x)$, and $P\hat\beta$ minimizes $\|y-XP\beta\|^2$: least squares on the symmetrized inputs $Px_i$. This is a hard constraint, not a trade-off. The antisymmetric part of $\beta$ earns nothing in the fit, because its contributions to an example and to its mirror image cancel on average, so any penalty on it drives it to zero.

The same computation works for any finite group $G$ of orthogonal matrices (shifts, rotations by multiples of $90°$, and so on) when each example is replaced by its whole orbit. Let $P=\frac1{|G|}\sum_{g\in G}g$. Then $P$ is the orthogonal projection onto the vectors fixed by every $g$, and
$$
\frac1{|G|}\sum_{g\in G}(y-\beta^\top gx)^2=(y-\beta^\top Px)^2+\frac1{|G|}\sum_{g\in G}\big(\beta^\top(g-P)x\big)^2 ,
$$
a mean plus a variance over the orbit. The variance term depends only on $(I-P)\beta$ and vanishes for invariant $\beta$. This is the linear-regression case of the view of [Chen, Dobriban and Lee (2020)](https://arxiv.org/abs/1907.10905), that augmentation is averaging over the orbits of a group.

*The cost.* If the truth has $\beta_Q\ne0$, the augmented estimator can never learn $\beta_Q^\top x$, however much data it sees. For mirror-symmetric inputs (part 5) the excess risk is at least $\beta_Q^\top\Sigma\beta_Q$. Augmentation is a bet that the symmetry holds: good for photographs of cats, bad for telling "b" from "d".

**5.** (a) Mirror symmetry of the inputs gives $S\Sigma S=\mathbb E[Sx\,(Sx)^\top]=\Sigma$, so $S\Sigma=\Sigma S$. Then $\Sigma$ commutes with $P$ and $Q$, and $P\Sigma Q=\Sigma PQ=0$.

(b) With $a=y-\beta_P^\top x$ and $c=\beta_Q^\top x$ as in 4, the augmented loss of one example averages to $(1-\alpha)(a-c)^2+\alpha(a+c)^2=a^2+c^2-2(1-2\alpha)\,ac$. Now $\mathbb E[ac]=\mathbb E[yx]^\top\beta_Q-\beta_P^\top\Sigma\beta_Q=\mathbb E[yx]^\top\beta_Q$ by (a). Writing $\rho=1-2\alpha$ and $h=\mathbb E[yx]$, the augmented risk is
$$
\mathbb E\big(y-\beta_P^\top x\big)^2+\beta_Q^\top\Sigma\beta_Q-2\rho\,h^\top\beta_Q .
$$
The clean risk is the case $\rho=1$. Both separate into a $\beta_P$ part, which is the same, and a $\beta_Q$ part. Minimizing over $\beta_Q$ in the range of $Q$ gives $Q\Sigma Q\,\beta_Q=\rho\,Qh$, so $\beta_Q=\rho\,\beta_Q^{\mathrm{clean}}$.

(c) Adding $\kappa\,\beta_Q^\top\Sigma\beta_Q$ to the clean risk gives $(1+\kappa)\,Q\Sigma Q\,\beta_Q=Qh$, so $\beta_Q=\beta_Q^{\mathrm{clean}}/(1+\kappa)$ and leaves $\beta_P$ alone. This matches (b) iff $1/(1+\kappa)=1-2\alpha$, i.e. $\kappa=2\alpha/(1-2\alpha)$. As $\alpha\to\tfrac12$, $\kappa\to\infty$ and the non-invariant part is removed entirely, which is part 4. Partial augmentation is a soft version of the same symmetry constraint: ridge on exactly the directions the symmetry says should not matter.
"""
    ),
]

CELLS = OPENER + L1 + NOISE
