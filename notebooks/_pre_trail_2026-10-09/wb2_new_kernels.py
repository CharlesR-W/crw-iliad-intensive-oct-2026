"""WB2 new material: kernels and the kernel trick (subsection 1.E)."""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
### 1.E. Kernels and the kernel trick

**The question.** Linear regression has a complete theory: closed-form
estimators, exact bias and variance, and regularization as a filter on the
data's principal directions. Can we keep all of that and still fit nonlinear
functions, even ones from an infinite-dimensional family? Yes: replace the
input $x$ by a feature vector $\phi(x)$ and do linear regression on the
features. The *kernel trick* is the observation that ridge regression on
features needs only the inner products $k(x,x')=\phi(x)^\top\phi(x')$, which
can often be computed cheaply even when $\phi(x)$ has astronomically many, or
infinitely many, coordinates.

**Why it matters here.** Kernel methods were the leading nonparametric
methods before deep learning. They are the cleanest setting in which a model
with more parameters than data points, fit to zero training error, can be
solved exactly: the ridgeless limit of kernel ridge regression is a
minimum-norm interpolant, the object behind double descent and benign
overfitting in section 3 of this workbook (3.B, 3.D). And they are the main
bridge from linear theory to neural networks: a network linearized around its
initialization is kernel regression with the *neural tangent kernel*, and a
network with a frozen random first layer approximates a kernel method
(end of this subsection). WB5 §1 derives scaling-law learning curves for ridge
regression with a power-law spectrum, which covers kernel and random-feature
models.

**What you get.** 1.E.1: ridge regression in dual form, the representer
theorem, and the ridgeless limit as a minimum-norm interpolant. 1.E.2: the
polynomial and Gaussian kernels, and random features. 1.E.3: Gaussian-process
regression, whose posterior mean is kernel ridge regression.

**Setup.** Training data are pairs $(x_1,y_1),\dots,(x_n,y_n)$. The inputs
$x_i$ lie in any set $\mathcal X$ (vectors in $\mathbb R^d$, but also strings
or graphs), and the labels form a vector $y\in\mathbb R^n$. A *feature map*
$\phi:\mathcal X\to\mathbb R^D$ sends each input to $D$ features, where $D$
may be much larger than $n$ or infinite (then $\mathbb R^D$ means the
square-summable sequences; nothing below changes). The model is linear in the
features, $f_w(x)=w^\top\phi(x)$. The feature design matrix
$\Phi\in\mathbb R^{n\times D}$ has rows $\phi(x_i)^\top$. Ridge regression on
the features is

$$
\hat w_\lambda=\arg\min_w\ \tfrac1n\|y-\Phi w\|^2+\lambda\|w\|^2=(\Phi^\top\Phi+n\lambda I_D)^{-1}\Phi^\top y ,\qquad \lambda>0 .
$$

The *kernel* of the feature map is $k(x,x')=\phi(x)^\top\phi(x')$. The
*kernel matrix* (Gram matrix) is $K=\Phi\Phi^\top\in\mathbb R^{n\times n}$,
with entries $K_{ij}=k(x_i,x_j)$. For a test input $x$, write
$k_x=\Phi\phi(x)=\big(k(x,x_1),\dots,k(x,x_n)\big)^\top$. Take the thin SVD
$\Phi=U\,\mathrm{diag}(s_1,\dots,s_r)\,V^\top$, where $r$ is the rank of
$\Phi$, the singular values are $s_k>0$, and the left and right singular
vectors are $u_k\in\mathbb R^n$ and $v_k\in\mathbb R^D$. Then
$K=\sum_ks_k^2u_ku_k^\top$: the nonzero eigenvalues of the kernel matrix are
the squared singular values of the feature design. In this basis ridge is a
filter (1.D.4.1, with $\Phi$ in place of the design matrix $X$): it multiplies
the least-squares coefficient along the $k$-th direction by
$s_k^2/(s_k^2+n\lambda)$, keeping directions with $s_k^2\gg n\lambda$ and
suppressing those with $s_k^2\ll n\lambda$.

**Which functions are kernels.** A symmetric function
$k:\mathcal X\times\mathcal X\to\mathbb R$ is a *positive-definite kernel* if
every Gram matrix $[k(z_a,z_b)]_{a,b\le N}$, for every finite set of points
$z_1,\dots,z_N$, is positive semidefinite. Any $k=\phi^\top\phi$ is one,
because $c^\top Kc=\|\Phi^\top c\|^2\ge0$. The converse is what makes kernels
useful (Moore–Aronszajn; stated, not proved): every positive-definite kernel
is an inner product of features in some Hilbert space. So one may pick $k$
directly, without ever writing down $\phi$. The functions the model can
represent, $f_w$, form the *reproducing-kernel Hilbert space* (RKHS)
$\mathcal H_k$, with norm $\|f\|_{\mathcal H}=\min\{\|w\|:f_w=f\}$. So "small
$\|w\|$" and "small RKHS norm" are the same penalty. Spectrally (Mercer's
theorem, stated): if the integral operator of $k$ under the input
distribution has eigenvalues $\mu_j$ and eigenfunctions $e_j$, then
$\|\sum_jc_je_j\|_{\mathcal H}^2=\sum_jc_j^2/\mu_j$. The norm charges heavily
for components along small-eigenvalue eigenfunctions, which for smooth kernels
such as the Gaussian kernel below are the rapidly varying ones. With many
training inputs, the top eigenvalues of $K/n$ approximate the top $\mu_j$.

#### 1.E.1. Ridge in dual form, the representer theorem, and the ridgeless limit

1. *Push-through.* For $a>0$, show that $(\Phi^\top\Phi+aI_D)^{-1}\Phi^\top=\Phi^\top(\Phi\Phi^\top+aI_n)^{-1}$. Deduce that $\hat w_\lambda=\Phi^\top\hat\alpha$ with $\hat\alpha=(K+n\lambda I_n)^{-1}y$, and that the prediction at any input $x$ is
$$\hat f_\lambda(x)=\sum_{i=1}^n\hat\alpha_i\,k(x,x_i)=k_x^\top(K+n\lambda I_n)^{-1}y .$$
Suppose evaluating $k$ costs $c$ operations. Compare the costs of fitting and of predicting in the primal form (a $D\times D$ solve) and the dual form (an $n\times n$ solve). When is each cheaper?
2. *Representer theorem.* Let $\mathcal L:\mathbb R^n\to\mathbb R$ be any loss that depends on the parameters only through the training predictions $\Phi w$, and minimize $J(w)=\mathcal L(\Phi w)+\lambda\|w\|^2$. Show that every minimizer has the form $\Phi^\top\alpha$ for some $\alpha\in\mathbb R^n$, and rewrite $J$ in terms of $\alpha$ and $K$ only. For the squared loss $\mathcal L(v)=\frac1n\|y-v\|^2$, check that $\hat\alpha$ from part 1 minimizes the rewritten objective.
3. *Ridgeless limit.* Assume $K$ is invertible. This needs $n\le D$; for the Gaussian kernel of 1.E.2 it holds whenever the $x_i$ are distinct. Show that as $\lambda\to0$, $\hat f_\lambda(x)\to k_x^\top K^{-1}y$, that this limit interpolates the training data, and that $\hat w_0=\Phi^\top K^{-1}y$ is the minimum-norm interpolant: it minimizes $\|w\|$ subject to $\Phi w=y$, with $\|\hat w_0\|^2=y^\top K^{-1}y$.
4. *What the interpolant pays for.* Show that the fitted values and the norm of the fit are
$$\hat y=\Phi\hat w_\lambda=\sum_k\frac{s_k^2}{s_k^2+n\lambda}\,(u_k^\top y)\,u_k,\qquad \|\hat w_\lambda\|^2=\sum_k\frac{s_k^2\,(u_k^\top y)^2}{(s_k^2+n\lambda)^2} .$$
In the setting of part 3, let $y_i=f^*(x_i)+\varepsilon_i$, where $f^*=f_{w^*}$ and the noise is independent with mean $0$ and variance $\sigma^2$. Show that $\mathbb E\|\hat w_0\|^2=\sum_k(v_k^\top w^*)^2+\sigma^2\,\mathrm{Tr}\,K^{-1}$. Which part of $y$ makes the interpolant's norm large when $K$ has tiny eigenvalues, as kernel matrices of smooth kernels do?

<details><summary>Hint</summary>

For 1, check that $\Phi^\top(\Phi\Phi^\top+aI)=(\Phi^\top\Phi+aI)\Phi^\top$, then multiply by inverses on each side. For 2, write $w=\Phi^\top\alpha+w_\perp$ with $w_\perp$ orthogonal to every $\phi(x_i)$, so $\Phi w_\perp=0$, and use Pythagoras. For 3, any interpolant is $\hat w_0+w'$ with $\Phi w'=0$; show that $w'\perp\hat w_0$. For 4, $K(K+aI)^{-1}=\sum_k\frac{s_k^2}{s_k^2+a}u_ku_k^\top$, and $f^*$ evaluated at the training inputs is $\Phi w^*$, whose component along $u_k$ is $s_k\,v_k^\top w^*$.
</details>
"""
    ),
    answer(
        r"""
**1.** Both sides of $\Phi^\top(\Phi\Phi^\top+aI_n)=\Phi^\top\Phi\Phi^\top+a\Phi^\top=(\Phi^\top\Phi+aI_D)\Phi^\top$ agree. For $a>0$ both bracketed matrices are positive definite, hence invertible. Multiplying on the left by $(\Phi^\top\Phi+aI_D)^{-1}$ and on the right by $(\Phi\Phi^\top+aI_n)^{-1}$ gives the identity. With $a=n\lambda$, $\hat w_\lambda=\Phi^\top(K+n\lambda I)^{-1}y=\Phi^\top\hat\alpha$, and $\hat f_\lambda(x)=\phi(x)^\top\Phi^\top\hat\alpha=k_x^\top\hat\alpha$.

Costs, counting arithmetic operations:

| | fit | predict one $x$ |
|---|---|---|
| primal | form $\Phi^\top\Phi$: $O(nD^2)$; solve: $O(D^3)$ | compute $\phi(x)$, then $O(D)$ |
| dual | form $K$: $O(n^2c)$; solve: $O(n^3)$ | $O(nc)$ |

The dual form wins when $D\gg n$, and it is the only option when $D=\infty$; the primal form wins when $n\gg D$. For large $n$ the $O(n^3)$ solve and the $O(n^2)$ memory for $K$ are the bottleneck of kernel methods, which is what random features (1.E.2.3) address.

**2.** Split $w=w_\parallel+w_\perp$, where $w_\parallel=\Phi^\top\alpha$ lies in the span of $\phi(x_1),\dots,\phi(x_n)$ and $w_\perp$ is orthogonal to it. Each entry of $\Phi w_\perp$ is $\phi(x_i)^\top w_\perp=0$, so $\mathcal L(\Phi w)=\mathcal L(\Phi w_\parallel)$, while $\|w\|^2=\|w_\parallel\|^2+\|w_\perp\|^2$. Hence $J(w)=J(w_\parallel)+\lambda\|w_\perp\|^2$, and any $w$ with $w_\perp\neq0$ is strictly beaten by $w_\parallel$. Substituting $\Phi w=\Phi\Phi^\top\alpha=K\alpha$ and $\|w\|^2=\alpha^\top K\alpha$,

$$
J=\mathcal L(K\alpha)+\lambda\,\alpha^\top K\alpha .
$$

The infinite-dimensional problem has become an $n$-dimensional one that needs only kernel values. The same proof works for any regularizer that is a nondecreasing function of $\|w\|$ (with "a minimizer" in place of "every minimizer" when it is not strictly increasing).

For the squared loss, the gradient in $\alpha$ is $-\frac2nK(y-K\alpha)+2\lambda K\alpha=\frac2nK\big[(K+n\lambda I)\alpha-y\big]$, which vanishes at $\hat\alpha$. The objective is a convex quadratic in $\alpha$, so a stationary point is a minimizer. (If $K$ is singular, other minimizers $\hat\alpha+v$ with $Kv=0$ exist, but $\|\Phi^\top v\|^2=v^\top Kv=0$, so they all give the same $w$.)

**3.** If $K$ is invertible, $(K+n\lambda I)^{-1}\to K^{-1}$, so $\hat f_\lambda(x)\to k_x^\top K^{-1}y$. At the training inputs, $\Phi\hat w_0=\Phi\Phi^\top K^{-1}y=y$, so the limit interpolates. Any other interpolant is $\hat w_0+w'$ with $\Phi w'=0$, and $\hat w_0^\top w'=y^\top K^{-1}\Phi w'=0$, so $\|\hat w_0+w'\|^2=\|\hat w_0\|^2+\|w'\|^2\ge\|\hat w_0\|^2$. Finally $\|\hat w_0\|^2=y^\top K^{-1}\Phi\Phi^\top K^{-1}y=y^\top K^{-1}y$. In RKHS language: among all functions in $\mathcal H_k$ that fit the data exactly, $k_x^\top K^{-1}y$ has the smallest RKHS norm. Here $\hat w_0=\Phi^+y$ (Moore–Penrose pseudoinverse): the same minimum-norm interpolant that gradient descent from zero finds in linear regression with more parameters than data (1.D.4.4, 3.B.1).

**4.** On the span of the $u_k$, $K(K+aI)^{-1}$ acts as $s_k^2/(s_k^2+a)$, and it is zero on the orthogonal complement. So $\hat y=K\hat\alpha=K(K+n\lambda I)^{-1}y$ is as stated, and $\|\hat w_\lambda\|^2=\hat\alpha^\top K\hat\alpha=\sum_ks_k^2(u_k^\top y)^2/(s_k^2+n\lambda)^2$. This is the primal ridge filter of 1.D.4, as it must be: it is the same estimator.

At $\lambda=0$, $\|\hat w_0\|^2=\sum_k(u_k^\top y)^2/s_k^2$. Write $y=\Phi w^*+\varepsilon$. Since $u_k^\top\Phi w^*=s_kv_k^\top w^*$, and $\mathbb E(u_k^\top\varepsilon)^2=\sigma^2$ in every direction,

$$
\mathbb E\|\hat w_0\|^2=\sum_k(v_k^\top w^*)^2+\sigma^2\sum_k\frac1{s_k^2}=\|P_{\rm row}w^*\|^2+\sigma^2\,\mathrm{Tr}\,K^{-1},
$$

where $P_{\rm row}$ projects onto the span of the $\phi(x_i)$. The signal contributes at most $\|w^*\|^2$, however small the eigenvalues. The noise contributes $\sigma^2/s_k^2$ per direction, which explodes along the smallest eigenvalues of $K$: the interpolant fits the noise using exactly the directions the kernel considers least plausible (largest RKHS cost), and pays for it in norm. Ridge suppresses those directions instead. Whether the interpolant's noise fit actually hurts test error depends on how many small eigenvalues there are and how they spread; that is benign overfitting, 3.D.
"""
    ),
    markdown(
        r"""
#### 1.E.2. Two kernels and random features

**Why.** The kernel trick pays off when $k$ is cheap and $\phi$ is huge. The
two standard kernels on $\mathbb R^d$ show both extremes: the *polynomial
kernel* $k(x,x')=(1+x^\top x')^m$ of degree $m\in\{1,2,\dots\}$, with
finitely but astronomically many features, and the *Gaussian* (or RBF) kernel
$k(x,x')=\exp\!\big(-\|x-x'\|^2/(2\ell^2)\big)$ with length scale $\ell>0$,
with infinitely many. Going the other way, random features approximate a
kernel by a finite, randomly drawn $\phi$, which is a two-layer network with
a frozen first layer.

The Gaussian kernel is *strictly* positive definite: its Gram matrix is
invertible for any set of distinct points (stated). So ridgeless Gaussian
kernel regression can interpolate any labels whatsoever, including pure
noise. When many inputs lie within a length scale of each other, though, $K$
has very small eigenvalues and is badly conditioned.

1. *Polynomial kernel.* Expand $(1+x^\top x')^m$ by the multinomial theorem and write it as $\phi(x)^\top\phi(x')$, where the coordinates of $\phi(x)$ are scaled monomials $\prod_ix_i^{j_i}$ of total degree at most $m$. Show that there are $\binom{d+m}{m}$ of them. Evaluate this for $d=784$ (the pixels of an MNIST digit) with $m=3$ and $m=5$, and compare with the $O(d)$ cost of evaluating the kernel.
2. *Gaussian kernel in one dimension.* Factor $e^{-(x-x')^2/(2\ell^2)}$ and Taylor-expand the cross term to show that $k(x,x')=\sum_{j=0}^\infty\phi_j(x)\phi_j(x')$ with
$$\phi_j(x)=e^{-x^2/(2\ell^2)}\,\frac{(x/\ell)^j}{\sqrt{j!}} .$$
So the Gaussian kernel is an inner product of infinitely many features. For $|x|\lesssim\ell$, how large is $\phi_j(x)$ for large $j$? Using $\|f\|_{\mathcal H}=\min\{\|w\|:f_w=f\}$, what does the RKHS norm charge for a function that needs a large degree-$j$ component?
3. *Random Fourier features* ([Rahimi and Recht, 2007](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)). Let $w\sim\mathcal N(0,\ell^{-2}I_d)$ and $b\sim\mathrm{Unif}[0,2\pi]$, independent. Show that
$$\mathbb E\big[2\cos(w^\top x+b)\cos(w^\top x'+b)\big]=e^{-\|x-x'\|^2/(2\ell^2)} .$$
Draw $m$ independent pairs $(w_j,b_j)$ and set $z(x)=\sqrt{2/m}\,\big(\cos(w_j^\top x+b_j)\big)_{j=1}^m$. Then $z(x)^\top z(x')$ is an average of $m$ independent unbiased estimates of the Gaussian kernel; each has variance at most $1$ (stated), so the error is at most about $1/\sqrt m$. For $n=10^6$ training points, compare the cost of ridge regression on $m=10^4$ random features with that of exact kernel ridge regression.

<details><summary>Hint</summary>

For 1, $(1+\sum_{i=1}^dx_ix_i')^m=\sum_{j_0+j_1+\dots+j_d=m}\frac{m!}{j_0!\,j_1!\cdots j_d!}\prod_{i\ge1}(x_ix_i')^{j_i}$; count the tuples $(j_0,\dots,j_d)$ by stars and bars. For 2, $-(x-x')^2=-x^2-x'^2+2xx'$. For 3, use $2\cos A\cos B=\cos(A-B)+\cos(A+B)$, note that $\mathbb E_b\cos(c+2b)=0$ for every constant $c$, and use the Gaussian characteristic function $\mathbb E\,e^{iw^\top\delta}=e^{-\|\delta\|^2/(2\ell^2)}$.
</details>
"""
    ),
    answer(
        r"""
**1.** By the multinomial theorem, with $j_0=m-\sum_{i\ge1}j_i$,

$$
(1+x^\top x')^m=\sum_{j_0+\dots+j_d=m}\frac{m!}{j_0!\,j_1!\cdots j_d!}\prod_{i=1}^d x_i^{j_i}\,x_i'^{\,j_i}=\phi(x)^\top\phi(x'),\qquad \phi_{(j_1,\dots,j_d)}(x)=\sqrt{\frac{m!}{j_0!\,j_1!\cdots j_d!}}\ \prod_{i=1}^dx_i^{j_i} .
$$

The coordinates are indexed by tuples $(j_0,\dots,j_d)$ of $d+1$ non-negative integers summing to $m$. Stars and bars gives $\binom{m+d}{d}=\binom{d+m}{m}$. For $d=784$: about $8.1\times10^7$ features at $m=3$ and $2.5\times10^{12}$ at $m=5$. The kernel costs one $784$-dimensional inner product and a power. So polynomial-kernel ridge is ridge over all polynomials of degree at most $m$ (each monomial weighted by the square root of its multinomial coefficient), at the price of an inner product.

**2.** $e^{-(x-x')^2/(2\ell^2)}=e^{-x^2/(2\ell^2)}\,e^{-x'^2/(2\ell^2)}\,e^{xx'/\ell^2}$, and $e^{xx'/\ell^2}=\sum_{j\ge0}\frac{(x/\ell)^j}{\sqrt{j!}}\cdot\frac{(x'/\ell)^j}{\sqrt{j!}}$. Multiplying in the two Gaussian factors gives the stated $\phi_j$. For $|x|\le\ell$, $|\phi_j(x)|\le1/\sqrt{j!}$, which is super-exponentially small. Every $f_w$ has the form $f_w(x)=e^{-x^2/(2\ell^2)}\sum_jw_j\,\frac{(x/\ell)^j}{\sqrt{j!}}$, so the coefficient of $x^j$ in the power series of $e^{x^2/(2\ell^2)}f_w(x)$ is $w_j/(\ell^j\sqrt{j!})$, and power series are unique, so this fixes $w$. Making that coefficient of order one costs $w_j^2\sim j!\,\ell^{2j}$: the RKHS norm grows super-exponentially with the degree a function needs. Gaussian-kernel ridge is therefore a strong smoothness prior, which charges enormously for variation on scales below $\ell$. (Example 16.2 of Shalev-Shwartz and Ben-David, reading below, does this expansion with $\ell=1$.)

**3.** Write $A=w^\top x+b$ and $B=w^\top x'+b$. Then $2\cos A\cos B=\cos\big(w^\top(x-x')\big)+\cos\big(w^\top(x+x')+2b\big)$. Averaging over $b$ with $w$ fixed kills the second term. The first is the real part of $e^{iw^\top\delta}$ with $\delta=x-x'$, and for $w\sim\mathcal N(0,\ell^{-2}I)$, $\mathbb E\,e^{iw^\top\delta}=e^{-\|\delta\|^2/(2\ell^2)}$. So $\mathbb E[2\cos A\cos B]=k(x,x')$, and $z(x)^\top z(x')=\frac1m\sum_j2\cos(w_j^\top x+b_j)\cos(w_j^\top x'+b_j)$ averages $m$ independent copies. (The single-term variance is $1-k^2+\tfrac12k^4\in[\tfrac12,1]$.)

Cost at $n=10^6$: exact kernel ridge needs $K$, which has $10^{12}$ entries (8 TB in double precision), and an $O(n^3)=10^{18}$ solve. Ridge on $m=10^4$ random features needs $Z^\top Z$ in $O(nm^2)=10^{14}$ operations (it can be accumulated in batches) and an $O(m^3)=10^{12}$ solve.

The map $x\mapsto z(x)$ is the hidden layer of a two-layer network with cosine activations and frozen random weights; ridge on $z$ trains only the readout, and as $m\to\infty$ it converges to Gaussian kernel ridge regression. For finite $m$ it is a model in its own right: in high-dimensional limits, ridgeless regression on $m$ random features has a test-error peak near $m\approx n$, like the interpolation peak that 3.A–3.C derive for linear least squares ([Mei and Montanari, 2019](https://arxiv.org/abs/1908.05355)). The same construction works for any continuous shift-invariant kernel with $k(0)=1$, sampling $w$ from its Fourier transform, which is a probability distribution by Bochner's theorem (stated; Theorem 1 of Rahimi and Recht).
"""
    ),
    markdown(
        r"""
#### 1.E.3. Gaussian-process regression is kernel ridge regression

**Why.** Ridge regression is the posterior mean of a Bayesian linear model
(1.D.1). The kernel version of that statement is Gaussian-process regression:
put a prior directly on functions, with the kernel as covariance, and the
posterior mean is kernel ridge regression. The Bayesian view also supplies
error bars, and shows what they cannot see.

A *Gaussian process* $f\sim\mathrm{GP}(0,k)$ is a random function such that,
for any finite set of inputs $z_1,\dots,z_N$, the vector
$\big(f(z_1),\dots,f(z_N)\big)$ is jointly Gaussian with mean $0$ and
covariance matrix $[k(z_a,z_b)]_{a,b}$. The kernel is now a covariance
function; positive definiteness is exactly the condition that makes every one
of these matrices a valid covariance. Notation as in the setup: $K$ is the
kernel matrix of the training inputs and $k_x=(k(x,x_1),\dots,k(x,x_n))^\top$.

1. *Weight-space view.* Take finite $D$, and let $f(x)=w^\top\phi(x)$ with prior $w\sim\mathcal N(0,\tau^2I_D)$, where $\tau>0$. Show that $f\sim\mathrm{GP}(0,\tau^2k)$. (For $D=\infty$ this construction fails, since independent $\mathcal N(0,\tau^2)$ coordinates have infinite norm; the GP is then defined directly as above, and its draws are almost surely not in the RKHS. The GP prior and the RKHS penalty give the same predictor but describe different function classes.)
2. *Posterior.* Observe $y_i=f(x_i)+\varepsilon_i$ with $\varepsilon\sim\mathcal N(0,\sigma^2I_n)$ independent of $f$. Show that, given $y$, $f(x)$ is Gaussian with
$$\mathbb E[f(x)\mid y]=k_x^\top\Big(K+\frac{\sigma^2}{\tau^2}I_n\Big)^{-1}y,\qquad \mathrm{Var}[f(x)\mid y]=\tau^2\Big(k(x,x)-k_x^\top\Big(K+\frac{\sigma^2}{\tau^2}I_n\Big)^{-1}k_x\Big).$$
Conclude that the posterior mean is the kernel ridge prediction with $n\lambda=\sigma^2/\tau^2$. For finite $D$, check the variance against the weight-space posterior of 1.D.1.1 (with $\Phi$ in place of $X$), $w\mid y\sim\mathcal N\big(\hat w_\lambda,\ \sigma^2(\Phi^\top\Phi+n\lambda I)^{-1}\big)$.
3. *Error bars.* The posterior variance does not depend on $y$. Show that it is at most the prior variance $\tau^2k(x,x)$; that at a training input it tends to $0$ as $\sigma\to0$ (assume $K$ invertible); and that for the Gaussian kernel it returns to the prior variance $\tau^2$ far from all training inputs. What can this error bar not detect?

<details><summary>Hint</summary>

For 2, use Gaussian conditioning: if $(a,b)$ is jointly Gaussian with mean zero, then $a\mid b\sim\mathcal N\big(\Sigma_{ab}\Sigma_{bb}^{-1}b,\ \Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\big)$. Here $a=f(x)$ and $b=y$. For the weight-space check, use $(\Phi^\top\Phi+cI)^{-1}=c^{-1}\big(I-\Phi^\top(K+cI)^{-1}\Phi\big)$, which follows from 1.E.1.1. For 3, at a training input $x_j$, $k_{x_j}$ is the $j$-th column of $K$.
</details>
"""
    ),
    answer(
        r"""
**1.** For inputs $z_1,\dots,z_N$, the vector $(f(z_a))_a=(\phi(z_a)^\top w)_a$ is a linear image of the Gaussian vector $w$, hence jointly Gaussian with mean zero. Its covariance is $\mathbb E[f(z_a)f(z_b)]=\phi(z_a)^\top\mathbb E[ww^\top]\phi(z_b)=\tau^2k(z_a,z_b)$.

**2.** $(f(x),y)$ is jointly Gaussian with $\mathrm{Cov}(y)=\tau^2K+\sigma^2I$, $\mathrm{Cov}(f(x),y)=\tau^2k_x^\top$ and $\mathrm{Var}\,f(x)=\tau^2k(x,x)$. Conditioning gives

$$
\mathbb E[f(x)\mid y]=\tau^2k_x^\top(\tau^2K+\sigma^2I)^{-1}y=k_x^\top\Big(K+\frac{\sigma^2}{\tau^2}I\Big)^{-1}y,
$$

$$
\mathrm{Var}[f(x)\mid y]=\tau^2k(x,x)-\tau^4k_x^\top(\tau^2K+\sigma^2I)^{-1}k_x=\tau^2\Big(k(x,x)-k_x^\top\Big(K+\frac{\sigma^2}{\tau^2}I\Big)^{-1}k_x\Big).
$$

The mean is $k_x^\top(K+n\lambda I)^{-1}y$ with $n\lambda=\sigma^2/\tau^2$, the same calibration as ridge-as-MAP in 1.D.1.1, as it should be: the two are the same model.

Weight-space check: write $c=n\lambda=\sigma^2/\tau^2$. The posterior variance of $f(x)=\phi(x)^\top w$ is

$$
\sigma^2\phi(x)^\top(\Phi^\top\Phi+cI)^{-1}\phi(x)=\frac{\sigma^2}{c}\Big(\phi(x)^\top\phi(x)-\phi(x)^\top\Phi^\top(K+cI)^{-1}\Phi\phi(x)\Big)=\tau^2\Big(k(x,x)-k_x^\top(K+cI)^{-1}k_x\Big).
$$

The identity in the hint holds because $(\Phi^\top\Phi+cI)\big(I-\Phi^\top(K+cI)^{-1}\Phi\big)=\Phi^\top\Phi+cI-\Phi^\top(K+cI)(K+cI)^{-1}\Phi=cI$.

The two traditions use the same predictor but tune it differently: kernel ridge usually picks $\lambda$ by cross-validation, Gaussian-process regression picks $\sigma,\tau$ and the kernel's own parameters (such as $\ell$) by maximizing the marginal likelihood of $y$.

**3.** $(K+cI)^{-1}$ is positive definite, so the subtracted term is non-negative and the posterior variance is at most $\tau^2k(x,x)$. At a training input, $k_{x_j}$ is the $j$-th column of $K$, so $K^{-1}k_{x_j}=\mathbf 1_j$, the $j$-th standard basis vector, and as $\sigma\to0$ the variance tends to $\tau^2(K_{jj}-K_{jj})=0$. For the Gaussian kernel, $k_x\to0$ as $x$ moves away from all training inputs, so the variance returns to $\tau^2k(x,x)=\tau^2$.

Because the variance does not depend on $y$, it cannot detect that the model is wrong: a kernel with the wrong length scale, the wrong noise level, or a target that is not a plausible draw from the prior gives equally confident error bars. They are calibrated only if the prior is right. In practice $y$ enters only through the hyperparameters fitted by marginal likelihood.
"""
    ),
    markdown(
        r"""
**Where this leads.** Linearize a network $f(x;\theta)$ around its
initialization $\theta_0$:
$f(x;\theta)\approx f(x;\theta_0)+\nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0)$.
This is a linear model with features $\phi(x)=\nabla_\theta f(x;\theta_0)$ and
kernel $\Theta(x,x')=\nabla_\theta f(x;\theta_0)^\top\nabla_\theta f(x';\theta_0)$,
the *neural tangent kernel*
([Jacot, Gabriel and Hongler, 2018](https://arxiv.org/abs/1806.07572)).
Gradient flow on squared loss for a linear model that can interpolate the data
converges to the interpolant closest to its starting point (1.D.4.4), so in
this linearized ("lazy" or "kernel") regime the trained network predicts
$f(x;\theta_0)$ plus ridgeless kernel regression of the residuals
$y_i-f(x_i;\theta_0)$ with kernel $\Theta$. Separately, a randomly initialized
fully connected network, with independent weights whose variance scales as one
over the layer width, becomes a Gaussian process as the widths go to infinity
([Lee et al., 2018](https://arxiv.org/abs/1711.00165)), so 1.E.3 also
describes exact Bayesian inference in infinitely wide networks. Much of the
debate about why deep networks generalize is about how far real networks are
from this kernel regime. The Training Dynamics materials take up the
linearized network and the lazy-versus-rich distinction; the infinite-width
NTK theory is on the Physics of Deep Learning day.
"""
        + references(
            book(
                "Understanding Machine Learning: From Theory to Algorithms",
                "http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning",
                "Shalev-Shwartz and Ben-David (2014; free PDF)",
                "ch. 16, §16.1–16.2",
            ),
            book(
                "Gaussian Processes for Machine Learning",
                "https://gaussianprocess.org/gpml/chapters/RW2.pdf",
                "Rasmussen and Williams (2006; free online)",
                "ch. 2, §2.1–2.3; ch. 6, §6.1–6.2",
            ),
        ),
    ),
]
