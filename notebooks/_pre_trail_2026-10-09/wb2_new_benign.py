"""WB2 new material: 3.D. Benign overfitting (a junk tail acts as ridge).

Goes in section 3 (Overfitting and double descent), after 3.C and before
3.E "What regime are we in?". Numerical checks: checks/wb2_new_benign_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
### 3.D. Benign overfitting: a junk tail acts as ridge

**The question.** Deep networks can fit their training data exactly, noisy
labels included, and still predict well. The usual intuition says that a fit
passing through every noisy point has absorbed the noise and must pay for it at
test time. In the simplest linear model that intuition is right. Take
$y=x^\top\beta+\varepsilon$ with all $p$ features i.i.d. $\mathcal N(0,1)$,
noise variance $\sigma^2$, and $n$ samples with $p>n+1$. The minimum-norm interpolant then
has expected excess risk $\|\beta\|^2(1-n/p)+\sigma^2n/(p-n-1)$ (3.B.1: squared bias plus
variance), which diverges as $p\downarrow n+1$ and tends to the risk of predicting zero
as $p\to\infty$. In that model interpolation can beat predicting zero (3.C.1), but at
every fixed ratio $p/n>1$ its excess risk stays bounded away from zero, however large $n$ is.
*Benign overfitting* is when an interpolant has zero training error on noisy
labels and an excess test risk close to the best achievable. When does that
happen?

**What you get.** An exactly solvable model in which you can see the mechanism.
Suppose the features beyond the useful ones form a *tail*: many directions, each
with small variance. Then the minimum-norm interpolant, restricted to the useful features,
is nearly ridge regression, with the tail supplying the penalty. The noise the tail
memorizes costs about (training residual)$/m$ at test time, where $m$ is the
tail dimension. Two conditions fall out: the tail must be high-dimensional
compared with $n$, and its total variance must be small compared with $n$. These
are the conditions of Bartlett, Long, Lugosi and Tsigler's characterization,
specialized to this model.

**Setup.**
- Features $x=(x_s,x_t)$, with *signal* $x_s\in\mathbb R^k$, $x_s\sim\mathcal N(0,I_k)$, and *tail* $x_t\in\mathbb R^m$, $x_t\sim\mathcal N(0,\nu I_m)$, independent of $x_s$. The per-coordinate tail variance $\nu$ is small. The tail's total variance is $c=m\nu=\mathbb E\|x_t\|^2$.
- Labels $y=x_s^\top\beta_s+\varepsilon$, with noise $\varepsilon\sim\mathcal N(0,\sigma^2)$ independent of $x$. The tail carries no signal. Write $r^2=\|\beta_s\|^2$.
- $n$ i.i.d. samples with $k<n<m$. Stack them as $X=[X_s,\,X_t]$, with $X_s\in\mathbb R^{n\times k}$, $X_t\in\mathbb R^{n\times m}$, and labels $y\in\mathbb R^n$.
- $X$ has more columns than rows, so many $b$ satisfy $Xb=y$. The *minimum-norm interpolant* is $\hat\beta=X^\top(XX^\top)^{-1}y$ ($XX^\top$ is invertible almost surely because $m\ge n$). It is the $\lambda\to0^+$ limit of ridge, and the point to which gradient descent from $0$ converges. Split it as $\hat\beta=(\hat\beta_s,\hat\beta_t)$.
- The *excess risk* of a coefficient vector $b=(b_s,b_t)$ is $\mathcal R(b)=\mathbb E\big[(x^\top b-x_s^\top\beta_s)^2\big]$ over a fresh $x$; it excludes the irreducible $\sigma^2$. Because $x_s$ and $x_t$ are independent with covariances $I_k$ and $\nu I_m$,
$$
\mathcal R(b)=\|b_s-\beta_s\|^2+\nu\|b_t\|^2 .
$$
- For comparison, *ridge on the signal features alone* is
$$
\hat\beta^{\rm R}_\lambda=\arg\min_{b\in\mathbb R^k}\ \tfrac1n\|y-X_sb\|^2+\lambda\|b\|^2=(X_s^\top X_s+n\lambda I_k)^{-1}X_s^\top y .
$$

#### 3.D.1. The tail supplies a ridge penalty

1. Show that $\hat\beta_s=X_s^\top(X_sX_s^\top+X_tX_t^\top)^{-1}y$ and $\hat\beta_t=X_t^\top(X_sX_s^\top+X_tX_t^\top)^{-1}y$. Prove the push-through identity $X_s^\top(X_sX_s^\top+cI_n)^{-1}=(X_s^\top X_s+cI_k)^{-1}X_s^\top$ for $c>0$. Conclude that if $X_tX_t^\top$ were exactly $cI_n$, then $\hat\beta_s$ would be $\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$.
2. Write the columns of $X_t$ as $\sqrt\nu\,z_j$, $j=1,\dots,m$, with $z_j\sim\mathcal N(0,I_n)$ i.i.d., so $X_tX_t^\top=\nu W$ with $W=\sum_jz_jz_j^\top$. Show that $\mathbb E\,W/m=I_n$, that the diagonal entries of $W/m$ have variance $2/m$ and the off-diagonal ones $1/m$. So $X_tX_t^\top\to cI_n$ entry by entry as $m\to\infty$ at fixed $c$, by the law of large numbers.
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
#### 3.D.2. What the memorized noise costs, and when overfitting is benign

Same setup: signal $x_s\sim\mathcal N(0,I_k)$, tail $x_t\sim\mathcal N(0,\nu I_m)$ with $c=m\nu$, labels $y=x_s^\top\beta_s+\varepsilon$, minimum-norm interpolant $\hat\beta=(\hat\beta_s,\hat\beta_t)=X^\top(XX^\top)^{-1}y$, excess risk $\mathcal R(\hat\beta)=\|\hat\beta_s-\beta_s\|^2+\nu\|\hat\beta_t\|^2$. Write $u=(XX^\top)^{-1}y\in\mathbb R^n$, so $\hat\beta_s=X_s^\top u$ and $\hat\beta_t=X_t^\top u$.

1. *Idealized tail.* Suppose $X_tX_t^\top=cI_n$ exactly. Show that $u=(y-X_s\hat\beta_s)/c$. Deduce that on the training inputs the tail's predictions $X_t\hat\beta_t$ are exactly the ridge residuals $y-X_s\hat\beta^{\rm R}_\lambda$ ($n\lambda=c$), and that the tail's contribution to the test risk is
$$
\nu\|\hat\beta_t\|^2=\frac{\|y-X_s\hat\beta^{\rm R}_\lambda\|^2}{m}.
$$
Why can the tail reproduce residuals of size $\sigma$ at every training point while contributing so little at a fresh point?
2. *Real tail.* Let $\mu_{\min},\mu_{\max}$ be the extreme eigenvalues of $X_tX_t^\top$. Show that $\nu\|\hat\beta_t\|^2\le\nu\mu_{\max}\|y\|^2/\mu_{\min}^2$. Using 3.D.1.4, conclude that $\nu\|\hat\beta_t\|^2\le\frac{\|y\|^2}{m}\cdot\frac{(1+d)^2}{(1-d)^4}$ with high probability, where $d\approx\sqrt{n/m}$. With $\mathbb E\|y\|^2=n(r^2+\sigma^2)$, the tail's cost is $O\big((r^2+\sigma^2)\,n/m\big)$, which tends to $0$ as $m\to\infty$ at fixed $c$.
3. *Pure-noise tail.* Take $k=0$, so $y=\varepsilon\sim\mathcal N(0,\sigma^2I_n)$ and $X=X_t$. Using the quoted fact $\mathbb E\,W^{-1}=I_n/(m-n-1)$ for $W=\sum_{j=1}^mz_jz_j^\top$, $z_j\sim\mathcal N(0,I_n)$ i.i.d., $m>n+1$, show that $\mathbb E\,\nu\|\hat\beta_t\|^2=\sigma^2n/(m-n-1)$. This does not depend on $\nu$. What happens as $m\downarrow n+1$? Is it the small per-coordinate variance $\nu$, or the dimension $m$, that makes memorizing noise cheap?
4. *Assemble.* Argue that as $m\to\infty$ at fixed $c$, $\hat\beta_s\to\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$, and hence $\mathcal R(\hat\beta)$ tends to the ridge risk $\|\hat\beta^{\rm R}_\lambda-\beta_s\|^2$, while the training error stays exactly zero. For $n\gg k$, use $X_s^\top X_s\approx nI_k$ to show that the ridge risk, averaged over the noise, is approximately
$$
\frac{c^2r^2+\sigma^2nk}{(n+c)^2}\approx\frac{\sigma^2k}{n}+\frac{c^2r^2}{n^2}\quad(c\ll n).
$$
Combine with 1 to write the total excess risk as roughly $\sigma^2k/n+c^2r^2/n^2+\sigma^2(n-k)/m$. State in words the two conditions under which interpolation is benign, i.e. the excess risk tends to $0$ as $n\to\infty$.
5. *Contrast with the isotropic model.* All $p$ features $\mathcal N(0,1)$ and $n<p$; the minimum-norm excess risk is $\|\beta\|^2(1-n/p)+\sigma^2n/(p-n-1)$ (quoted, for $p>n+1$). Think of the features beyond the first few as a tail with $\nu=1$. Which of the two conditions fails near $p\approx n$, and which fails for $p\gg n$? Why can no choice of $p$ satisfy both?
6. *Effective ranks.* For a covariance $\Sigma$ with eigenvalues $\lambda_1\ge\lambda_2\ge\dots$, Bartlett et al. define $r_j(\Sigma)=\sum_{i>j}\lambda_i/\lambda_{j+1}$ and $R_j(\Sigma)=\big(\sum_{i>j}\lambda_i\big)^2/\sum_{i>j}\lambda_i^2$, and $k^*=\min\{j\ge0:r_j(\Sigma)\ge bn\}$ for a constant $b>1$. They call a sequence of problems benign when $r_0(\Sigma)/n\to0$, $k^*/n\to0$ and $n/R_{k^*}(\Sigma)\to0$. Compute these for this model (assume $\nu<1$, $k+c<bn\le m$) and check that the three conditions are the ones you found in 4.

<details><summary>Hint</summary>

For 1, write out $(X_sX_s^\top+cI_n)u=y$ and recognize $X_s^\top u$. For the tail cost, $\|\hat\beta_t\|^2=u^\top X_tX_t^\top u$. For 2, $\lambda_{\min}(XX^\top)\ge\mu_{\min}$ because $X_sX_s^\top\succeq0$, so $\|u\|\le\|y\|/\mu_{\min}$. For 3, average over $\varepsilon$ first: $\mathbb E\,\varepsilon^\top A\varepsilon=\sigma^2\mathrm{Tr}A$. For 4, a ridge estimate with $X_s^\top X_s=nI_k$ shrinks each coordinate of the signal-only least-squares estimate $(X_s^\top X_s)^{-1}X_s^\top y$ by $n/(n+c)$; add the squared bias and the variance. For the convergence of $\hat\beta_s$, write $X_tX_t^\top=c(I+\Delta)$ with $\|\Delta\|\to0$ and use $\|(A+E)^{-1}-A^{-1}\|\le\|A^{-1}\|\,\|E\|\,\|(A+E)^{-1}\|$.
</details>
"""
    ),
    answer(
        r"""
**1.** $u$ solves $(X_sX_s^\top+cI_n)u=y$, i.e. $cu=y-X_s(X_s^\top u)=y-X_s\hat\beta_s$. By 3.D.1.1, $\hat\beta_s=\hat\beta^{\rm R}_\lambda$ with $n\lambda=c$. On the training inputs the tail predicts $X_t\hat\beta_t=X_tX_t^\top u=cu=y-X_s\hat\beta^{\rm R}_\lambda$: the signal block makes the ridge fit, and the tail adds exactly the ridge residual, so the total fit interpolates. The tail's test cost is
$$
\nu\|\hat\beta_t\|^2=\nu\,u^\top X_tX_t^\top u=\nu c\|u\|^2=\frac{\nu}{c}\|y-X_s\hat\beta^{\rm R}_\lambda\|^2=\frac{\|y-X_s\hat\beta^{\rm R}_\lambda\|^2}{m}.
$$
For $n\gg k$, the expected squared ridge residual is about $\sigma^2(n-k)+c^2r^2/n$ (noise the signal block cannot fit, plus the shrinkage bias), so for $c\ll n$ the tail costs about $\sigma^2(n-k)/m$.

Why it is cheap: $\hat\beta_t=\sum_iu_i\,x_{t,i}$ is a combination of the $n$ training tail vectors. These are nearly orthogonal in $\mathbb R^m$ (here exactly orthogonal, each of squared norm $c$), so $x_{t,i}^\top\hat\beta_t=cu_i$ reproduces residual $i$ at training point $i$. A fresh $x_t$ is nearly orthogonal to all of them, and $x_t^\top\hat\beta_t$ has variance $\nu\|\hat\beta_t\|^2$. The fitted function is the ridge fit plus a narrow spike at each training point, and the spikes occupy a vanishing fraction of the $m$-dimensional tail.

**2.** $\nu\|\hat\beta_t\|^2=\nu\,u^\top X_tX_t^\top u\le\nu\mu_{\max}\|u\|^2$. Because $XX^\top\succeq X_tX_t^\top$, $\lambda_{\min}(XX^\top)\ge\mu_{\min}$, so $\|u\|=\|(XX^\top)^{-1}y\|\le\|y\|/\mu_{\min}$. On the event of 3.D.1.4, $\mu_{\min}\ge c(1-d)^2$ and $\mu_{\max}\le c(1+d)^2$, so
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
    markdown(
        r"""
The same "the tail acts as ridge" mechanism is why kernel "ridgeless" regression
on high-dimensional inputs can generalize: the many small eigenvalues of the
kernel matrix act as an implicit ridge penalty
([Liang and Rakhlin 2020](https://arxiv.org/abs/1808.00387)).
"""
        + reading(
            "Benign overfitting in linear regression",
            "https://arxiv.org/abs/1906.11300",
            "Peter L. Bartlett, Philip M. Long, Gábor Lugosi, Alexander Tsigler (PNAS, 2020)",
            "Non-asymptotic upper and lower bounds on the excess risk of the minimum-norm interpolant, for "
            "any covariance, in terms of two effective ranks of the covariance. Interpolation is "
            "near-optimal only if the number of directions unimportant for prediction greatly exceeds n "
            "while their total variance stays small compared with n.",
            "Read §2–§3: Definition 3 (effective ranks), Theorem 4 (the bounds), and Theorem 6, whose "
            "part 2 (a few strong directions plus small isotropic noise in p ≫ n dimensions) is the model "
            "of 3.D with decaying rather than equal signal eigenvalues. §4 discusses the relevance to "
            "deep networks. The proofs in §5 are long and can be skipped.",
            importance=4, length=3,
        )
    ),
]
