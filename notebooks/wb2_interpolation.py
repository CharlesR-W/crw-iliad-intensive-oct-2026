"""MDL1 §5: interpolation without harm (kernel ridge and the ridgeless limit; benign overfitting).

Formulas are checked by checks/wb2_new_kernels_check.py and checks/wb2_new_benign_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
<a id="wb2-section-5"></a>

## 5. Interpolation without harm: kernels and benign overfitting

When a fit that passes through every noisy training point is still close to
optimal: kernel "ridgeless" regression is a minimum-norm interpolant, and many
weak directions act on it as an implicit ridge penalty.

Notation: $\lambda$ is the ridge penalty and $a=n\lambda$; in 5.1, $k(\cdot,\cdot)$ is the kernel, $K$ the $n\times n$ kernel matrix and $s_k$ the singular values of the feature matrix; in 5.2–5.3, $k$ is the number of signal directions, $m$ the number of tail directions, $\nu$ their variance and $c=m\nu$ the tail's total variance, which acts as $n\lambda$; $\mu_{\min}(\cdot)$, $\mu_{\max}(\cdot)$ and $\mu_i$ denote eigenvalues; $\sigma^2$ is the noise variance and $r^2=\|\beta\|^2$.
"""
        + reading(
            "Benign overfitting in linear regression",
            "https://arxiv.org/abs/1906.11300",
            "Peter L. Bartlett, Philip M. Long, Gábor Lugosi, Alexander Tsigler (PNAS, 2020)",
            "Interpolation is near-optimal exactly when the directions unimportant for prediction are many "
            "compared with n but small in total variance; 5.2–5.3 is a solvable case. Read §2–§3.",
            importance=5, length=3,
        )
        + supplemental()
        + reading(
            "Neural tangent kernel",
            "https://en.wikipedia.org/wiki/Neural_tangent_kernel",
            "Wikipedia",
            "A wide network linearized around its initialization trains as kernel regression with the gradient "
            "features; in that regime training ends at the ridgeless kernel interpolant of 5.1.",
            importance=3, length=2,
        )
        + reading(
            "Double descent",
            "https://en.wikipedia.org/wiki/Double_descent",
            "Wikipedia",
            "The test-error curve that peaks at the interpolation threshold and falls again; 1.3–1.6 compute it "
            "for least squares and 5.2–5.3 say when the right branch is benign.",
            importance=2, length=2,
        )
        + r"""
Also: [Jacot, Gabriel and Hongler (2018)](https://arxiv.org/abs/1806.07572), the neural tangent kernel paper; [Liang and Rakhlin (2020)](https://arxiv.org/abs/1808.00387), for high-dimensional inputs the many small eigenvalues of the kernel matrix act as an implicit ridge, the kernel version of 5.2; [Lee et al. (2018)](https://arxiv.org/abs/1711.00165), wide random networks are Gaussian processes; [Rahimi and Recht (2007)](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html), random Fourier features; [Mei and Montanari (2019)](https://arxiv.org/abs/1908.05355), double descent in random-feature regression.
"""
        + references(
            book("Understanding Machine Learning: From Theory to Algorithms",
                 "http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning",
                 "Shalev-Shwartz and Ben-David (2014; free PDF)", "ch. 16, §16.1–16.2"),
            book("Gaussian Processes for Machine Learning", "https://gaussianprocess.org/gpml/chapters/RW2.pdf",
                 "Rasmussen and Williams (2006; free online)", "ch. 2, §2.1–2.3"),
        )
        + r"""
### Exercises: kernel ridge regression and the ridgeless limit

**5.1. Dual form, representer theorem, ridgeless limit.** Training data are
$(x_1,y_1),\dots,(x_n,y_n)$ with labels $y\in\mathbb R^n$. A feature map
$\phi:\mathcal X\to\mathbb R^D$ may have $D\gg n$ or $D=\infty$. The model is
$f_w(x)=w^\top\phi(x)$, the feature design $\Phi\in\mathbb R^{n\times D}$ has
rows $\phi(x_i)^\top$, and ridge on the features is
$\hat w_\lambda=\arg\min_w\frac1n\|y-\Phi w\|^2+\lambda\|w\|^2=(\Phi^\top\Phi+n\lambda I_D)^{-1}\Phi^\top y$.
The *kernel* is $k(x,x')=\phi(x)^\top\phi(x')$, the *kernel matrix*
$K=\Phi\Phi^\top$ has entries $k(x_i,x_j)$, and
$k_x=\big(k(x,x_1),\dots,k(x,x_n)\big)^\top$. The thin SVD is
$\Phi=\sum_ks_ku_kv_k^\top$, so $K=\sum_ks_k^2u_ku_k^\top$. The Gaussian kernel
is $k(x,x')=e^{-\|x-x'\|^2/(2\ell^2)}$.

1. *Push-through.* For $a>0$, show that $(\Phi^\top\Phi+aI_D)^{-1}\Phi^\top=\Phi^\top(\Phi\Phi^\top+aI_n)^{-1}$. Deduce that $\hat w_\lambda=\Phi^\top\hat\alpha$ with $\hat\alpha=(K+n\lambda I_n)^{-1}y$, and that the prediction at any input $x$ is
$$\hat f_\lambda(x)=\sum_{i=1}^n\hat\alpha_i\,k(x,x_i)=k_x^\top(K+n\lambda I_n)^{-1}y .$$
Suppose each kernel evaluation costs a fixed number of operations. Compare the costs of fitting and of predicting in the primal form (a $D\times D$ solve) and the dual form (an $n\times n$ solve). When is each cheaper?
2. *Representer theorem.* Let $\mathcal L:\mathbb R^n\to\mathbb R$ be any loss that depends on the parameters only through the training predictions $\Phi w$, and minimize $J(w)=\mathcal L(\Phi w)+\lambda\|w\|^2$. Show that every minimizer has the form $\Phi^\top\alpha$ for some $\alpha\in\mathbb R^n$, and rewrite $J$ in terms of $\alpha$ and $K$ only. For the squared loss $\mathcal L(v)=\frac1n\|y-v\|^2$, check that $\hat\alpha$ from part 1 minimizes the rewritten objective.
3. *Ridgeless limit.* Assume $K$ is invertible. This needs $n\le D$; for the Gaussian kernel it holds whenever the $x_i$ are distinct. Show that as $\lambda\to0$, $\hat f_\lambda(x)\to k_x^\top K^{-1}y$, that this limit interpolates the training data, and that $\hat w_0=\Phi^\top K^{-1}y$ is the minimum-norm interpolant: it minimizes $\|w\|$ subject to $\Phi w=y$, with $\|\hat w_0\|^2=y^\top K^{-1}y$.
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

The dual form wins when $D\gg n$, and it is the only option when $D=\infty$; the primal form wins when $n\gg D$. For large $n$ the $O(n^3)$ solve and the $O(n^2)$ memory for $K$ are the bottleneck of kernel methods, which is what random features (Rahimi and Recht, in the readings above) address.

**2.** Split $w=w_\parallel+w_\perp$, where $w_\parallel=\Phi^\top\alpha$ lies in the span of $\phi(x_1),\dots,\phi(x_n)$ and $w_\perp$ is orthogonal to it. Each entry of $\Phi w_\perp$ is $\phi(x_i)^\top w_\perp=0$, so $\mathcal L(\Phi w)=\mathcal L(\Phi w_\parallel)$, while $\|w\|^2=\|w_\parallel\|^2+\|w_\perp\|^2$. Hence $J(w)=J(w_\parallel)+\lambda\|w_\perp\|^2$, and any $w$ with $w_\perp\neq0$ is strictly beaten by $w_\parallel$. Substituting $\Phi w=\Phi\Phi^\top\alpha=K\alpha$ and $\|w\|^2=\alpha^\top K\alpha$,

$$
J=\mathcal L(K\alpha)+\lambda\,\alpha^\top K\alpha .
$$

The infinite-dimensional problem has become an $n$-dimensional one that needs only kernel values. The same proof works for any regularizer that is a nondecreasing function of $\|w\|$ (with "a minimizer" in place of "every minimizer" when it is not strictly increasing).

For the squared loss, the gradient in $\alpha$ is $-\frac2nK(y-K\alpha)+2\lambda K\alpha=\frac2nK\big[(K+n\lambda I)\alpha-y\big]$, which vanishes at $\hat\alpha$. The objective is a convex quadratic in $\alpha$, so a stationary point is a minimizer. (If $K$ is singular, other minimizers $\hat\alpha+v$ with $Kv=0$ exist, but $\|\Phi^\top v\|^2=v^\top Kv=0$, so they all give the same $w$.)

**3.** If $K$ is invertible, $(K+n\lambda I)^{-1}\to K^{-1}$, so $\hat f_\lambda(x)\to k_x^\top K^{-1}y$. At the training inputs, $\Phi\hat w_0=\Phi\Phi^\top K^{-1}y=y$, so the limit interpolates. Any other interpolant is $\hat w_0+w'$ with $\Phi w'=0$, and $\hat w_0^\top w'=y^\top K^{-1}\Phi w'=0$, so $\|\hat w_0+w'\|^2=\|\hat w_0\|^2+\|w'\|^2\ge\|\hat w_0\|^2$. Finally $\|\hat w_0\|^2=y^\top K^{-1}\Phi\Phi^\top K^{-1}y=y^\top K^{-1}y$. In RKHS language: among all functions in $\mathcal H_k$ that fit the data exactly, $k_x^\top K^{-1}y$ has the smallest RKHS norm. Here $\hat w_0=\Phi^+y$ (Moore–Penrose pseudoinverse): the same minimum-norm interpolant that gradient descent from zero finds in linear regression with more parameters than data (4.3.4, 1.4).

**4.** On the span of the $u_k$, $K(K+aI)^{-1}$ acts as $s_k^2/(s_k^2+a)$, and it is zero on the orthogonal complement. So $\hat y=K\hat\alpha=K(K+n\lambda I)^{-1}y$ is as stated, and $\|\hat w_\lambda\|^2=\hat\alpha^\top K\hat\alpha=\sum_ks_k^2(u_k^\top y)^2/(s_k^2+n\lambda)^2$. This is the primal ridge filter of 4.3, as it must be: it is the same estimator.

At $\lambda=0$, $\|\hat w_0\|^2=\sum_k(u_k^\top y)^2/s_k^2$. Write $y=\Phi w^*+\varepsilon$. Since $u_k^\top\Phi w^*=s_kv_k^\top w^*$, and $\mathbb E(u_k^\top\varepsilon)^2=\sigma^2$ in every direction,

$$
\mathbb E\|\hat w_0\|^2=\sum_k(v_k^\top w^*)^2+\sigma^2\sum_k\frac1{s_k^2}=\|P_{\rm row}w^*\|^2+\sigma^2\,\mathrm{Tr}\,K^{-1},
$$

where $P_{\rm row}$ projects onto the span of the $\phi(x_i)$. The signal contributes at most $\|w^*\|^2$, however small the eigenvalues. The noise contributes $\sigma^2/s_k^2$ per direction, which explodes along the smallest eigenvalues of $K$: the interpolant fits the noise using exactly the directions the kernel considers least plausible (largest RKHS cost), and pays for it in norm. Ridge suppresses those directions instead. Whether the interpolant's noise fit actually hurts test error depends on how many small eigenvalues there are and how they spread; that is benign overfitting, 5.2–5.3.
"""
    ),
    markdown(
        r"""
<a id="wb2-benign"></a>

### Exercises: benign overfitting

In the isotropic model of 1.4, the minimum-norm interpolant's excess risk
stays bounded away from zero at every fixed $p/n>1$. *Benign overfitting* is
when an interpolant of noisy labels has excess risk close to the best
achievable. Here is an exactly solvable model in which a high-dimensional,
low-variance tail of features makes that happen.

**Setup for 5.2–5.3.**
- Features $x=(x_s,x_t)$, with *signal* $x_s\in\mathbb R^k$, $x_s\sim\mathcal N(0,I_k)$, and *tail* $x_t\in\mathbb R^m$, $x_t\sim\mathcal N(0,\nu I_m)$, independent of $x_s$. The tail's total variance is $c=m\nu=\mathbb E\|x_t\|^2$.
- Labels $y=x_s^\top\beta_s+\varepsilon$, with noise $\varepsilon\sim\mathcal N(0,\sigma^2)$ independent of $x$. The tail carries no signal. Write $r^2=\|\beta_s\|^2$.
- $n$ i.i.d. samples with $k<n<m$, stacked as $X=[X_s,\,X_t]$ with $X_s\in\mathbb R^{n\times k}$, $X_t\in\mathbb R^{n\times m}$, and labels $y\in\mathbb R^n$.
- The *minimum-norm interpolant* is $\hat\beta=X^\top(XX^\top)^{-1}y$, the $\lambda\to0^+$ limit of ridge and the limit of gradient descent from $0$. Split it as $\hat\beta=(\hat\beta_s,\hat\beta_t)$.
- The *excess risk* of $b=(b_s,b_t)$ over a fresh $x$ is $\mathcal R(b)=\mathbb E\big[(x^\top b-x_s^\top\beta_s)^2\big]=\|b_s-\beta_s\|^2+\nu\|b_t\|^2$.
- *Ridge on the signal features alone* is $\hat\beta^{\rm R}_\lambda=(X_s^\top X_s+n\lambda I_k)^{-1}X_s^\top y$.

**5.2. The tail supplies a ridge penalty.**

1. Show that $\hat\beta_s=X_s^\top(X_sX_s^\top+X_tX_t^\top)^{-1}y$ and $\hat\beta_t=X_t^\top(X_sX_s^\top+X_tX_t^\top)^{-1}y$. Prove the push-through identity $X_s^\top(X_sX_s^\top+cI_n)^{-1}=(X_s^\top X_s+cI_k)^{-1}X_s^\top$ for $c>0$. Conclude that if $X_tX_t^\top$ were exactly $cI_n$, then $\hat\beta_s$ would be $\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$.
2. Write the columns of $X_t$ as $\sqrt\nu\,z_j$, $j=1,\dots,m$, with $z_j\sim\mathcal N(0,I_n)$ i.i.d., so $X_tX_t^\top=\nu W$ with $W=\sum_jz_jz_j^\top$. Show that $\mathbb E\,W/m=I_n$, that the diagonal entries of $W/m$ have variance $2/m$ and the off-diagonal ones $1/m$. So $X_tX_t^\top\to cI_n$ entry by entry as $m\to\infty$ at fixed $c$.
3. Entry by entry is not enough, because there are $n^2$ entries. For a fixed unit vector $v\in\mathbb R^n$, show that $\mathbb E\|(W/m-I_n)v\|^2=(n+1)/m$. So $W/m$ moves a typical vector by about $\sqrt{n/m}$, and $X_tX_t^\top\approx cI_n$ needs $m\gg n$, not just $m$ large.
4. *Quoted* (Davidson–Szarek; [Vershynin's notes](https://arxiv.org/abs/1011.3027), Corollary 5.35): with probability at least $1-2e^{-t^2/2}$, every singular value of the $n\times m$ matrix $[z_1,\dots,z_m]$ lies in $[\sqrt m-\sqrt n-t,\ \sqrt m+\sqrt n+t]$. With $d=\sqrt{n/m}+t/\sqrt m<1$, deduce that every eigenvalue of $X_tX_t^\top$ lies in $[c(1-d)^2,\ c(1+d)^2]$, and that the smallest eigenvalue of $XX^\top$ is at least $c(1-d)^2$. Why does this lower bound matter, given that the variance of the minimum-norm interpolant involves $(XX^\top)^{-1}$?

<details><summary>Hint</summary>

For 1, split $X^\top=\begin{pmatrix}X_s^\top\\ X_t^\top\end{pmatrix}$ and note $XX^\top=X_sX_s^\top+X_tX_t^\top$. For the identity, start from $(X_s^\top X_s+cI_k)X_s^\top=X_s^\top(X_sX_s^\top+cI_n)$. For 2, $(W/m)_{il}=\frac1m\sum_jz_{ij}z_{lj}$ is an average of $m$ i.i.d. terms; $\mathbb Ez^4=3$ for a standard normal. For 3, by rotation invariance take $v=e_1$, so $(W/m)v=\frac1m\sum_jz_jz_{1j}$. Expand $\mathbb E\|(W/m)v\|^2$ as a double sum over $j,l$ and separate $j=l$ from $j\ne l$. For 4, the eigenvalues of $X_tX_t^\top$ are $\nu$ times the squared singular values; $X_sX_s^\top\succeq0$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\hat\beta=X^\top(XX^\top)^{-1}y$ with $X^\top=\begin{pmatrix}X_s^\top\\ X_t^\top\end{pmatrix}$, so the two blocks are $X_s^\top(XX^\top)^{-1}y$ and $X_t^\top(XX^\top)^{-1}y$, and $XX^\top=X_sX_s^\top+X_tX_t^\top$. For the identity, both sides of $(X_s^\top X_s+cI_k)X_s^\top=X_s^\top X_sX_s^\top+cX_s^\top=X_s^\top(X_sX_s^\top+cI_n)$ are equal. Multiply on the left by $(X_s^\top X_s+cI_k)^{-1}$ and on the right by $(X_sX_s^\top+cI_n)^{-1}$; both inverses exist because $c>0$. If $X_tX_t^\top=cI_n$, then
$$
\hat\beta_s=X_s^\top(X_sX_s^\top+cI_n)^{-1}y=(X_s^\top X_s+cI_k)^{-1}X_s^\top y=\hat\beta^{\rm R}_{\lambda},\qquad n\lambda=c .
$$
The interpolant never sees a penalty. The tail's Gram matrix plays the role of $n\lambda I_n$ inside the "kernel" form of ridge.

**2.** $\mathbb E\,z_jz_j^\top=I_n$, so $\mathbb E W/m=I_n$. A diagonal entry $\frac1m\sum_jz_{ij}^2$ has variance $\frac1m\mathrm{Var}(z^2)=\frac1m(3-1)=\frac2m$. An off-diagonal entry $\frac1m\sum_jz_{ij}z_{lj}$ ($i\ne l$) has mean $0$ and variance $\frac1m\mathbb E z_{ij}^2z_{lj}^2=\frac1m$. Each entry is an average of $m$ i.i.d. terms, so it converges to the corresponding entry of $I_n$, and $X_tX_t^\top=\nu W=c\,W/m\to cI_n$.

**3.** With $v=e_1$, $(W/m)v=\frac1m\sum_jz_jz_{1j}$, and
$$
\mathbb E\|(W/m)v\|^2=\frac1{m^2}\sum_{j,l}\mathbb E\big[z_{1j}z_{1l}\,z_j^\top z_l\big].
$$
For $j\ne l$ the factors are independent and $\mathbb E[z_{1j}z_j]=e_1$, so each of the $m(m-1)$ terms is $e_1^\top e_1=1$. For $j=l$, $\mathbb E\big[z_{1j}^2\|z_j\|^2\big]=\mathbb Ez_1^4+(n-1)=n+2$, for $m$ terms. So $\mathbb E\|(W/m)v\|^2=1+(n+1)/m$. Since $\mathbb E\,v^\top(W/m)v=1$,
$$
\mathbb E\|(W/m-I)v\|^2=\mathbb E\|(W/m)v\|^2-2\,\mathbb E\,v^\top(W/m)v+1=\frac{n+1}{m}.
$$
Summing over $v=e_1,\dots,e_n$ gives $\mathbb E\|W/m-I\|_F^2=n(n+1)/m$, which matches 2: $n\cdot\frac2m+n(n-1)\cdot\frac1m$. The individual entries are $O(1/\sqrt m)$, but their effect on a vector adds up over $n$ coordinates. The operator norm $\|W/m-I\|$ is about $2\sqrt{n/m}$ (part 4), so the approximation $X_tX_t^\top\approx cI_n$ is controlled by $n/m$.

**4.** The eigenvalues of $W$ are the squared singular values, so on the stated event they lie in $[(\sqrt m-\sqrt n-t)^2,(\sqrt m+\sqrt n+t)^2]=m[(1-d)^2,(1+d)^2]$. Multiplying by $\nu$ and using $m\nu=c$ puts the eigenvalues of $X_tX_t^\top$ in $[c(1-d)^2,c(1+d)^2]$. Since $X_sX_s^\top\succeq0$, $XX^\top\succeq X_tX_t^\top$, and so $\lambda_{\min}(XX^\top)\ge c(1-d)^2$, equivalently $\|(XX^\top)^{-1}\|\le1/\big(c(1-d)^2\big)$. (With $n=100$, $m=20000$ the edges are near $(1\pm0.07)^2\approx0.86,\,1.15$, and a simulation agrees.)

The noise enters the interpolant as $X^\top(XX^\top)^{-1}\varepsilon$, so its size is governed by the small eigenvalues of $XX^\top$. In the isotropic model with $p\approx n$, $XX^\top$ has eigenvalues near zero, and that is where the divergence of the risk at $p=n$ comes from. A high-dimensional tail puts a floor of about $c$ under every eigenvalue of $XX^\top$, which is exactly what an explicit ridge penalty $n\lambda=c$ does.
"""
    ),
    markdown(
        r"""
**5.3. What the memorized noise costs, and when overfitting is benign.** Same
setup. Write $u=(XX^\top)^{-1}y\in\mathbb R^n$, so $\hat\beta_s=X_s^\top u$ and
$\hat\beta_t=X_t^\top u$.

1. *Idealized tail.* Suppose $X_tX_t^\top=cI_n$ exactly. Show that $u=(y-X_s\hat\beta_s)/c$. Deduce that on the training inputs the tail's predictions $X_t\hat\beta_t$ are exactly the ridge residuals $y-X_s\hat\beta^{\rm R}_\lambda$ ($n\lambda=c$), and that the tail's contribution to the test risk is
$$
\nu\|\hat\beta_t\|^2=\frac{\|y-X_s\hat\beta^{\rm R}_\lambda\|^2}{m}.
$$
Why can the tail reproduce residuals of size $\sigma$ at every training point while contributing so little at a fresh point?
2. *Real tail.* Let $\mu_{\min},\mu_{\max}$ be the extreme eigenvalues of $X_tX_t^\top$. Show that $\nu\|\hat\beta_t\|^2\le\nu\mu_{\max}\|y\|^2/\mu_{\min}^2$. Using 5.2.4, conclude that $\nu\|\hat\beta_t\|^2\le\frac{\|y\|^2}{m}\cdot\frac{(1+d)^2}{(1-d)^4}$ with high probability, where $d\approx\sqrt{n/m}$. With $\mathbb E\|y\|^2=n(r^2+\sigma^2)$, the tail's cost is $O\big((r^2+\sigma^2)\,n/m\big)$, which tends to $0$ as $m\to\infty$ at fixed $c$.
3. *Pure-noise tail.* Take $k=0$, so $y=\varepsilon\sim\mathcal N(0,\sigma^2I_n)$ and $X=X_t$. Using the quoted fact $\mathbb E\,W^{-1}=I_n/(m-n-1)$ for $W=\sum_{j=1}^mz_jz_j^\top$, $z_j\sim\mathcal N(0,I_n)$ i.i.d., $m>n+1$, show that $\mathbb E\,\nu\|\hat\beta_t\|^2=\sigma^2n/(m-n-1)$. This does not depend on $\nu$. What happens as $m\downarrow n+1$? Is it the small per-coordinate variance $\nu$, or the dimension $m$, that makes memorizing noise cheap?
4. *Assemble.* Argue that as $m\to\infty$ at fixed $c$, $\hat\beta_s\to\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$, and hence $\mathcal R(\hat\beta)$ tends to the ridge risk $\|\hat\beta^{\rm R}_\lambda-\beta_s\|^2$, while the training error stays exactly zero. For $n\gg k$, use $X_s^\top X_s\approx nI_k$ to show that the ridge risk, averaged over the noise, is approximately
$$
\frac{c^2r^2+\sigma^2nk}{(n+c)^2}\approx\frac{\sigma^2k}{n}+\frac{c^2r^2}{n^2}\quad(c\ll n).
$$
Combine with 1 to write the total excess risk as roughly $\sigma^2k/n+c^2r^2/n^2+\sigma^2(n-k)/m$. State in words the two conditions under which interpolation is benign, i.e. the excess risk tends to $0$ as $n\to\infty$.
5. *Contrast with the isotropic model.* All $p$ features $\mathcal N(0,1)$ and $n<p$; the minimum-norm excess risk is $\|\beta\|^2(1-n/p)+\sigma^2n/(p-n-1)$ (1.4, for $p>n+1$). Think of the features beyond the first few as a tail with $\nu=1$. Which of the two conditions fails near $p\approx n$, and which fails for $p\gg n$? Why can no choice of $p$ satisfy both?
6. *Effective ranks.* For a covariance $\Sigma$ with eigenvalues $\mu_1\ge\mu_2\ge\dots$, Bartlett et al. define $r_j(\Sigma)=\sum_{i>j}\mu_i/\mu_{j+1}$ and $R_j(\Sigma)=\big(\sum_{i>j}\mu_i\big)^2/\sum_{i>j}\mu_i^2$, and $k^*=\min\{j\ge0:r_j(\Sigma)\ge bn\}$ for a constant $b>1$. They call a sequence of problems benign when $r_0(\Sigma)/n\to0$, $k^*/n\to0$ and $n/R_{k^*}(\Sigma)\to0$. Compute these for this model (assume $\nu<1$, $k+c<bn\le m$) and check that the three conditions are the ones you found in 4.

<details><summary>Hint</summary>

For 1, write out $(X_sX_s^\top+cI_n)u=y$ and recognize $X_s^\top u$. For the tail cost, $\|\hat\beta_t\|^2=u^\top X_tX_t^\top u$. For 2, $\mu_{\min}(XX^\top)\ge\mu_{\min}$ because $X_sX_s^\top\succeq0$, so $\|u\|\le\|y\|/\mu_{\min}$. For 3, average over $\varepsilon$ first: $\mathbb E\,\varepsilon^\top A\varepsilon=\sigma^2\mathrm{Tr}A$. For 4, a ridge estimate with $X_s^\top X_s=nI_k$ shrinks each coordinate of the signal-only least-squares estimate $(X_s^\top X_s)^{-1}X_s^\top y$ by $n/(n+c)$; add the squared bias and the variance. For the convergence of $\hat\beta_s$, write $X_tX_t^\top=c(I+\Delta)$ with $\|\Delta\|\to0$ and use $\|(A+E)^{-1}-A^{-1}\|\le\|A^{-1}\|\,\|E\|\,\|(A+E)^{-1}\|$.
</details>
"""
    ),
    answer(
        r"""
**1.** $u$ solves $(X_sX_s^\top+cI_n)u=y$, i.e. $cu=y-X_s(X_s^\top u)=y-X_s\hat\beta_s$. By 5.2.1, $\hat\beta_s=\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$. On the training inputs the tail predicts $X_t\hat\beta_t=X_tX_t^\top u=cu=y-X_s\hat\beta^{\rm R}_\lambda$: the signal block makes the ridge fit, and the tail adds exactly the ridge residual, so the total fit interpolates. The tail's test cost is
$$
\nu\|\hat\beta_t\|^2=\nu\,u^\top X_tX_t^\top u=\nu c\|u\|^2=\frac{\nu}{c}\|y-X_s\hat\beta^{\rm R}_\lambda\|^2=\frac{\|y-X_s\hat\beta^{\rm R}_\lambda\|^2}{m}.
$$
For $n\gg k$, the expected squared ridge residual is about $\sigma^2(n-k)+c^2r^2/n$ (noise the signal block cannot fit, plus the shrinkage bias), so for $c\ll n$ the tail costs about $\sigma^2(n-k)/m$.

Why it is cheap: $\hat\beta_t=\sum_iu_i\,x_{t,i}$ is a combination of the $n$ training tail vectors. These are nearly orthogonal in $\mathbb R^m$ (here exactly orthogonal, each of squared norm $c$), so $x_{t,i}^\top\hat\beta_t=cu_i$ reproduces residual $i$ at training point $i$. A fresh $x_t$ is nearly orthogonal to all of them, and $x_t^\top\hat\beta_t$ has variance $\nu\|\hat\beta_t\|^2$. The fitted function is the ridge fit plus a narrow spike at each training point, and the spikes occupy a vanishing fraction of the $m$-dimensional tail.

**2.** $\nu\|\hat\beta_t\|^2=\nu\,u^\top X_tX_t^\top u\le\nu\mu_{\max}\|u\|^2$. Because $XX^\top\succeq X_tX_t^\top$, $\mu_{\min}(XX^\top)\ge\mu_{\min}$, so $\|u\|=\|(XX^\top)^{-1}y\|\le\|y\|/\mu_{\min}$. On the event of 5.2.4, $\mu_{\min}\ge c(1-d)^2$ and $\mu_{\max}\le c(1+d)^2$, so
$$
\nu\|\hat\beta_t\|^2\le\frac{\nu c(1+d)^2}{c^2(1-d)^4}\|y\|^2=\frac{\|y\|^2}{m}\cdot\frac{(1+d)^2}{(1-d)^4}.
$$
With $\|y\|^2\approx n(r^2+\sigma^2)$ this is $O\big((r^2+\sigma^2)n/m\big)$. The bound uses $\|y\|^2$ where 1 had the smaller ridge residual. In simulation ($k=5$, $n=100$, $m=20000$, $\sigma=1$, $r^2=5$) the tail cost matches $\|y-X_s\hat\beta^{\rm R}_\lambda\|^2/m\approx0.005$ to within 1%, against a bound of about $0.05$.

**3.** With $k=0$, $\nu\|\hat\beta_t\|^2=\nu\,\varepsilon^\top(X_tX_t^\top)^{-1}X_tX_t^\top(X_tX_t^\top)^{-1}\varepsilon=\nu\,\varepsilon^\top(\nu W)^{-1}\varepsilon=\varepsilon^\top W^{-1}\varepsilon$. Averaging over $\varepsilon$ and then over $W$,
$$
\mathbb E\,\nu\|\hat\beta_t\|^2=\sigma^2\,\mathbb E\,\mathrm{Tr}\,W^{-1}=\frac{\sigma^2n}{m-n-1}.
$$
The $\nu$ cancels: rescaling the tail rescales $\hat\beta_t$ inversely, and the test cost is scale-free. As $m\downarrow n+1$ the cost diverges; this is the double-descent peak again, now inside the tail. For $m\gg n$ it is about $\sigma^2n/m$. So what makes memorization cheap is the *dimension* of the tail relative to $n$. A small $\nu$ is needed for a different reason (part 4): to keep the total $c=m\nu$, and hence the implicit penalty, small.

**4.** Write $X_tX_t^\top=c(I+\Delta)$, where $\|\Delta\|\le(1+d)^2-1\approx2\sqrt{n/m}\to0$, and $A=X_sX_s^\top+cI_n\succeq cI_n$. Then $\hat\beta_s-\hat\beta^{\rm R}_\lambda=X_s^\top\big[(A+c\Delta)^{-1}-A^{-1}\big]y$, and $\|(A+c\Delta)^{-1}-A^{-1}\|\le\frac1c\cdot c\|\Delta\|\cdot\frac1{c(1-\|\Delta\|)}\to0$. So $\hat\beta_s\to\hat\beta^{\rm R}_\lambda$. With 2, $\mathcal R(\hat\beta)=\|\hat\beta_s-\beta_s\|^2+\nu\|\hat\beta_t\|^2\to\|\hat\beta^{\rm R}_\lambda-\beta_s\|^2$. The training error is $0$ for every $m$, since $\hat\beta$ interpolates.

Ridge risk: if $X_s^\top X_s=nI_k$, then $\hat\beta^{\rm R}_\lambda=\frac{1}{n+c}X_s^\top y=\frac{n}{n+c}\beta_s+\frac1{n+c}X_s^\top\varepsilon$. The bias is $-\frac{c}{n+c}\beta_s$, and the noise term has covariance $\frac{\sigma^2n}{(n+c)^2}I_k$. Adding squared bias and variance,
$$
\mathbb E_\varepsilon\|\hat\beta^{\rm R}_\lambda-\beta_s\|^2=\frac{c^2r^2+\sigma^2nk}{(n+c)^2}.
$$
For $c\ll n$ this is $\approx\sigma^2k/n+c^2r^2/n^2$. (The fluctuations of $X_s^\top X_s$ around $nI_k$ are of relative size $\sqrt{k/n}$ and change this by a factor $1+O(k/n)$ on the variance term.) In total,
$$
\mathcal R(\hat\beta)\approx\underbrace{\frac{\sigma^2k}{n}}_{\text{signal variance}}+\underbrace{\frac{c^2r^2}{n^2}}_{\text{tail-as-ridge bias}}+\underbrace{\frac{\sigma^2(n-k)}{m}}_{\text{memorized noise}} .
$$
In simulation with $k=5$, $n=100$, $m=20000$: for $c=10$, $0.086$ against ridge's $0.082$ (and $\sigma^2k/n=0.05$); for $c=50$, both $\approx0.62$, dominated by the bias. At $m=200$ (still $c=10$) the risk is about $1.0$, of which $0.92$ is the tail term: a tail only twice as wide as $n$ is not benign.

Interpolation is benign ($\mathcal R\to0$ as $n\to\infty$) when, besides $k\ll n$:
- **The tail is high-dimensional:** $m\gg n$. The tail then absorbs the noise in nearly orthogonal directions at a cost $\approx\sigma^2n/m$, and its Gram matrix is close to a multiple of the identity, so it acts as an isotropic ridge.
- **The tail is small in total:** $c=m\nu\ll n$, i.e. the implicit penalty $\lambda=c/n$ is small compared with the signal eigenvalue $1$. Otherwise the tail-as-ridge over-shrinks the signal.

Both together require $\nu\ll n/m\ll1$: very many, very weak directions.

**5.** In the isotropic model every direction has variance $1$, so a tail of $m\approx p$ features has $c\approx p>n$. Near $p\approx n$ the tail is not high-dimensional: $\sigma^2n/(p-n-1)$ is part 3's formula, and it diverges. For $p\gg n$ the tail is high-dimensional, but its total variance $c\approx p\gg n$ makes the implicit ridge so strong that the fit captures only the fraction $n/p$ of $\beta$; the risk tends to $\|\beta\|^2(1-n/p)\to\|\beta\|^2$, the risk of predicting zero. Because $\nu=1$ is fixed, $m\gg n$ forces $c=m\gg n$, so the two conditions cannot hold together. (In the isotropic model the signal is also spread over all directions, so this "tail" is not junk; the analogy is about the variance and shrinkage mechanisms, not an exact match.)

**6.** The eigenvalues are $1$ ($k$ times) then $\nu$ ($m$ times). $r_0=(k+m\nu)/1=k+c$. For $j<k$, $r_j=k-j+c<bn$, and $r_k=m\nu/\nu=m\ge bn$, so $k^*=k$. Then $R_k=(m\nu)^2/(m\nu^2)=m$. The benign conditions become $(k+c)/n\to0$, $k/n\to0$ and $n/m\to0$: small tail total ($c\ll n$), few signal directions, and a high-dimensional tail ($m\gg n$). Their variance bound $\sigma^2(k^*/n+n/R_{k^*})=\sigma^2(k/n+n/m)$ matches the first and third terms in 4. For a non-isotropic tail, $r_k$ and $R_k$ are the right notions of "how many dimensions the tail effectively has".
"""
    ),
]
