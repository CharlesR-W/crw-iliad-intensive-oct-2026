#!/usr/bin/env python3
"""WB2 cells, part B: §6–§9. Concatenated after wb2_cells_a.CELLS by build_wb2.py.

Formulas used in the solutions are checked numerically in checks/wb2b_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    # ------------------------------------------------------------------ §6
    markdown(
        r"""
<a id="wb2-section-6"></a>

## 6. Information bottleneck and the information plane

**The result.** For a deterministic network, $I(X;T)=H(T)$ when the input is discrete;
it equals $H(X)$ while the layer is injective on the input support. If the
layer output has an atomless distribution, the mutual information is infinite. The "compression phase" seen in information-plane
plots of an injective deterministic layer therefore reflects the estimator. A binned estimate is the
entropy of a quantization, and saturating units shrink it. Once noise is added
to the layer, $I(X;T)$ becomes finite, and in the two-class model below it falls
when within-class covariance shrinks at fixed class means and noise scale.
6.F.i asks what these derivations do and do not settle about the published
accounts.

**Notation for this section.** $X$ is the input, $Y$ the label, and
$T=f(X)$ the activations of one hidden layer. $T$ is random because $X$ is.
$H(\cdot)$ is the entropy of a discrete variable and $h(\cdot)$ the
differential entropy of a density, both in nats. Mutual information is
$I(A;B)=H(A)-H(A\mid B)$, or the same with $h$. Because $T$ depends on $Y$ only
through $X$, the variables form a Markov chain $Y\to X\to T$, and the data
processing inequality gives $I(T;Y)\le I(X;Y)$.

### 6.A. Three accounts of the information plane

The *information bottleneck* (IB), introduced by
[Tishby, Pereira and Bialek (1999)](https://arxiv.org/abs/physics/0004057),
asks for a representation $T$ of $X$ that is as compressed as possible while
staying informative about $Y$:

$$
\min_{p(t\mid x)}\ I(X;T)-\beta\,I(T;Y),\qquad \beta>0 .
$$

The *information plane* plots each layer of a network as the point
$\big(I(X;T),\,I(T;Y)\big)$ and follows it through training.

**Tishby's claim** (Shwartz-Ziv and Tishby 2017). Training has two phases. In
a short *fitting* phase both coordinates rise. In a long *compression* phase
$I(X;T)$ falls while $I(T;Y)$ holds. The compression is driven by the
noise-dominated, diffusive part of SGD, and it is why networks generalize.
Their estimator binned the activations of a $\tanh$ network into 30 equal bins
on $[-1,1]$, with $X$ uniform over 4096 inputs.

**The critique** (Saxe et al. 2018). Compression appears with double-sided
saturating units like $\tanh$, and not with ReLU or linear units. Networks that
do not compress still generalize. The same curves appear under full-batch
gradient descent, so SGD noise is not their cause. There is also a conceptual
problem, which 6.B makes precise: for a deterministic network the quantity
being plotted is constant or infinite.

**A third reading** (Goldfeld et al. 2019). Add a
little Gaussian noise to each layer, so that $I(X;T)$ is finite and depends on
the weights. Then compression does occur, and it is driven by the clustering
of same-class representations. The binned estimates track the same clustering.
"""
    ),
    markdown(
        r"""
### 6.B. A deterministic network has nothing to compress

#### 6.B.i. Constant or infinite

1. Let $X$ be discrete and $T=f(X)$ for a deterministic $f$. Show that $I(X;T)=H(T)$, and that $I(X;T)=H(X)$ if $f$ is injective on the support of $X$.
2. In Shwartz-Ziv and Tishby's setup, $X$ is uniform over 4096 inputs. What is $I(X;T)$ for any layer that maps distinct inputs to distinct activation vectors? With 32-bit floats, can two distinct inputs still be mapped to exactly the same activation vector, and what would cause it?
3. Now let $X$ have a density on $\mathbb R^d$, and suppose $T=f(X)$ has an atomless distribution, i.e. $\Pr[T=t]=0$ for every $t$ (true, for example, if $f$ is smooth and injective). Show that $I(X;T)=\infty$.
4. Conclude: which changes to the weights can change $I(X;T)$ in cases 1 and 3?

<details><summary>Hint</summary>

For 3, let $q_\Delta$ quantize $T$ to cubes of side $\Delta$. Since $q_\Delta(T)$ is a function of $X$, part 1 and data processing give $I(X;T)\ge I(X;q_\Delta(T))=H(q_\Delta(T))$. Then use $H(Q)\ge-\log\max_k\Pr[Q=k]$.
</details>
"""
    ),
    answer(
        r"""
**1.** $I(X;T)=H(T)-H(T\mid X)$, and $H(T\mid X)=0$ because $T$ is determined by $X$. So $I(X;T)=H(T)=H(f(X))$. If $f$ is injective on the support, $f(X)$ is a relabelling of $X$, so $H(f(X))=H(X)$.

**2.** $H(X)=\log4096=12$ bits $\approx8.32$ nats, for every such layer, at every step of training. This value remains constant while the mapping stays injective. Float precision alone does not determine collision probabilities: dead ReLUs, finite-precision saturation, and other noninjective maps can produce exact collisions. Changes in a binned estimate need not reflect changes in this unquantized information.

**3.** Let $q_\Delta$ map $T$ to the cube of side $\Delta$ that contains it. $q_\Delta(T)$ is a deterministic function of $X$, so by 1, $I(X;q_\Delta(T))=H(q_\Delta(T))$. Since $X\to T\to q_\Delta(T)$, data processing gives $I(X;T)\ge H(q_\Delta(T))\ge-\log\max_k\Pr[q_\Delta(T)=k]$.

It remains to show that $m(\Delta):=\sup_{\text{cubes }Q\text{ of side }\Delta}\Pr[T\in Q]\to0$. Suppose not: then there are $\epsilon>0$ and cubes $Q_j$ with sides $\Delta_j\to0$ and $\Pr[T\in Q_j]\ge\epsilon$. By tightness, the $Q_j$ eventually lie in a compact set, so a subsequence of their centres converges to some $t_0$. Every ball around $t_0$ then has probability at least $\epsilon$, so $\Pr[T=t_0]\ge\epsilon$, contradicting atomlessness. Hence $H(q_\Delta(T))\to\infty$ and $I(X;T)=\infty$.

(Equivalently, when $T$ has a density, $h(T\mid X)=-\infty$ because $T$ given $X$ is a point mass.)

**4.** In case 3, none: $I(X;T)=\infty$ for every weight setting that keeps $T$ atomless. In case 1, only exact collisions, where two distinct inputs are mapped to exactly the same activation vector, can lower $H(f(X))$. Such collisions are possible, for example with dead ReLUs. The constancy conclusion requires injectivity on the discrete input support; the infinite-information conclusion requires atomlessness. Under those conditions, movement in a finite plotted estimate does not describe a change in the exact mutual information.
"""
    ),
    markdown(
        r"""
### 6.C. What the binned estimator measures

The binned estimator replaces $T$ by $\hat T=\mathrm{bin}(T)$, its bin index,
and reports $I(X;\hat T)$. Since $\hat T$ is a function of $X$, 6.B.i.1 says
this is just $H(\mathrm{bin}(T))$, the entropy of a quantization.

#### 6.C.i. Saturation looks like compression

Take one unit with standard normal pre-activation $Z$ and pre-activation scale
$a>0$, the weight norm. Training typically grows $a$.

1. $T=\tanh(aZ)$, binned into $K$ equal bins on $[-1,1]$, with $K$ even. Show that $H(\mathrm{bin}(T))\to\log2$ as $a\to\infty$. What is the limit as $a\to0$? How does it change if $K$ is odd?
2. Sketch $H(\mathrm{bin}(T))$ as a function of $a$ for $K=30$. Explain why steadily growing weights alone produce a "fitting phase" followed by a "compression phase".
3. $T=\mathrm{ReLU}(aZ)=\max(0,aZ)$, binned with a fixed width $\Delta$: $[0,\Delta),[\Delta,2\Delta),\dots$. Show that as $a\to\infty$,
$$
H(\mathrm{bin}(T))=\log2+\tfrac12\Big(h(|Z|)+\log\frac a\Delta\Big)+o(1),\qquad h(|Z|)=\tfrac12\log\frac{\pi e}2 .
$$
So the estimate grows without bound, like $\tfrac12\log a$.
4. Many implementations instead bin ReLU activations into $K$ equal bins on $[0,\max T]$, the observed range. What happens to the estimate as $a$ grows?
5. In one sentence: what quantity is the binned estimator measuring?

<details><summary>Hint</summary>

For 1, the bin edges are $e_j=-1+2j/K$; compute $\Pr[T\ge1-2/K]$. For 3, condition on the sign of $Z$, and use the standard fact that for a nice density, $H(\mathrm{bin}_\delta(W))=h(W)-\log\delta+o(1)$ as $\delta\to0$. Here the effective bin width for $Z$ is $\delta=\Delta/a$. For 4, ReLU is positively homogeneous.
</details>
"""
    ),
    answer(
        r"""
**1.** The top bin is $[1-2/K,1]$, and
$\Pr[T\ge1-2/K]=\Pr\big[Z\ge\operatorname{artanh}(1-2/K)/a\big]\to\Pr[Z>0]=\tfrac12$
as $a\to\infty$. The bottom bin likewise tends to $\tfrac12$, so every other
bin tends to $0$. Entropy is continuous on the finite simplex, so
$H\to\log2$: the unit has become a sign function, and the estimate counts
two cells.

As $a\to0$, $T\approx aZ$ concentrates at $0$. With $K$ even, $0$ is a bin
edge, so the two bins adjacent to $0$ each get mass $\to\tfrac12$ and
$H\to\log2$ again. With $K$ odd, the middle bin straddles $0$, takes all the
mass, and $H\to0$.

**2.** For $K=30$, exact bin probabilities give (from `checks/wb2b_check.py`):

| $a$ | 0.1 | 0.5 | 0.9 | 2 | 5 | 20 | 100 | $10^4$ |
|---|---|---|---|---|---|---|---|---|
| $H(\mathrm{bin}\tanh(aZ))$ (nats) | 1.83 | 3.21 | 3.40 (max) | 2.88 | 1.92 | 1.10 | 0.80 | 0.69 |

At small $a$ the activations occupy a few central bins. As $a$ grows they
spread over more bins, so the estimate rises ("fitting"). Past
$a\approx1$ they pile up in the two saturated end bins, so it falls toward
$\log2$ ("compression"). Monotone weight growth produces the two-phase curve
with no change in what the network distinguishes: $T$ stays an injective
function of $Z$ throughout.

**3.** Let $S=\mathbf 1[Z>0]$. All mass with $Z\le0$ sits at $T=0$, in bin
$0$, together with $\Pr[0<Z<\Delta/a]\to0$. Up to that vanishing overlap,
$S$ is a function of the bin, so
$$
H(\mathrm{bin}\,T)=H(S)+\Pr[S=1]\,H(\mathrm{bin}\,T\mid S=1)+o(1)
=\log2+\tfrac12H\big(\mathrm{bin}_{\Delta/a}(|Z|)\big)+o(1).
$$
Given $Z>0$, $Z$ has the density of $|Z|$, and binning $aZ$ at width $\Delta$
is binning $Z$ at width $\delta=\Delta/a\to0$. So
$H(\mathrm{bin}_\delta|Z|)=h(|Z|)-\log\delta+o(1)$. The half-normal has
$h(|Z|)=\tfrac12\log(2\pi e)-\log2=\tfrac12\log(\pi e/2)$, which gives the
formula. Check: with $\Delta=1/15$ and $a=100$ the formula gives $4.713$
nats and the exact value is $4.710$; at $a=1000$, $5.864$ against $5.864$.
The ReLU estimate keeps growing, with no compression phase. This is the
pattern Saxe et al. report.

**4.** $\max(0,aZ)/\max_i(0,aZ_i)$ does not depend on $a$, so the binned
estimate is exactly constant in $a$. The answer to 3 versus 4 is decided by a
choice of bins, not by the network.

**5.** It measures how many cells of size $\Delta$ the data occupy, i.e. the
spread of the representation measured at a resolution fixed by the analyst.
That is a geometric quantity. For a fixed binning rule it is a property of the distribution of $T$,
but it is not generally equal to the unquantized $I(X;T)$. The distribution
and geometry of $T$ can change even while the latter stays constant.
"""
    ),
    markdown(
        r"""
### 6.D. With noise, compression is clustering

Goldfeld et al. make the mutual information meaningful by adding noise:
$T=f(X)+\sigma\varepsilon$ with $\varepsilon\sim\mathcal N(0,I_d)$ independent
of $X$. Now $h(T\mid X)=h(\sigma\varepsilon)=\frac d2\log(2\pi e\sigma^2)$ is
finite, and $I(X;T)=h(T)-\frac d2\log(2\pi e\sigma^2)$ depends on $f$.
Differential entropy and the Gaussian channel are in Cover and Thomas,
*Elements of Information Theory*, ch. 8–9.

#### 6.D.i. The Gaussian channel

1. Suppose $f(X)\sim\mathcal N(\mu,S)$. Show that $I(X;T)=\tfrac12\log\det(I+S/\sigma^2)=\sum_k\tfrac12\log(1+s_k/\sigma^2)$, where $s_k$ are the eigenvalues of $S$.
2. Show that for any $f(X)$ with covariance $S$, $I(X;T)\le\tfrac12\log\det(I+S/\sigma^2)$.
3. Interpret $\tfrac12\log(1+s_k/\sigma^2)$ in terms of the number of distinguishable cells of size $\sigma$ along the $k$-th principal direction. Compare with 6.C.i.5.

<details><summary>Hint</summary>

$h(T\mid X)=h(\sigma\varepsilon)$ is the same for every $f$. For 2, among all densities with a given covariance, the Gaussian has the largest differential entropy.
</details>

#### 6.D.ii. A two-class model

Let $Y\in\{\pm1\}$ be uniform, and let $X=(Y,U)$ with $U\sim\mathcal N(0,I_d)$
independent of $Y$, so the label is a function of the input. The layer computes
$$
f(X)=Y\,m+W^{1/2}U ,
$$
so $m\in\mathbb R^d$ sets the class means $\pm m$ (between-class covariance
$mm^\top$) and $W$ is the within-class covariance. Then $T=f(X)+\sigma\varepsilon$.

1. Show that $I(X;T)=I(T;Y)+I(X;T\mid Y)$, and that $I(X;T\mid Y)=\tfrac12\log\det(I+W/\sigma^2)$.
2. Let $\Sigma_T=W+\sigma^2I$ and $d^2=m^\top\Sigma_T^{-1}m$. Show that $I(T;Y)$ depends on the model only through $d^2$, and that
$$
I(T;Y)=\log2-\mathbb E\log\!\big(1+e^{-L}\big),\qquad L\sim\mathcal N(2d^2,\,4d^2).
$$
Show that this increases with $d^2$ and tends to $\log2$.
3. Now shrink the within-class covariance, $W\to\alpha W$ with $\alpha\downarrow0$, keeping $m$ and $\sigma$ fixed. Describe the path of the point $\big(I(X;T),I(T;Y)\big)$. What are its limits?
4. What happens to $I(X;T)$ if you rescale the whole representation, $f\to cf$, and leave $\sigma$ fixed? What does this say about the role of $\sigma$?

<details><summary>Hint</summary>

For 1, use the chain rule on $I(X,Y;T)$ and the fact that $Y$ is a function of $X$. Given $Y$, $f(X)$ is Gaussian, so 6.D.i.1 applies. For 2, the posterior log-odds $L=\log\frac{p(T\mid Y=+1)}{p(T\mid Y=-1)}$ is linear in $T$. For monotonicity, show that a channel with smaller $d^2$ is the channel with larger $d^2$ followed by extra noise.
</details>
"""
    ),
    answer(
        r"""
**6.D.i.1.** $T\sim\mathcal N(\mu,S+\sigma^2I)$, so
$h(T)=\tfrac12\log\det\big(2\pi e(S+\sigma^2I)\big)$ and
$h(T\mid X)=\tfrac12\log\det(2\pi e\sigma^2I)$. Subtracting gives
$\tfrac12\log\det(I+S/\sigma^2)$. Diagonalizing $S$ gives the sum.

**6.D.i.2.** $h(T\mid X)$ is the same for every $f$. $T$ has covariance
$S+\sigma^2I$, and the Gaussian maximizes differential entropy at fixed
covariance, so $h(T)\le\tfrac12\log\det(2\pi e(S+\sigma^2I))$.

**6.D.i.3.** $\tfrac12\log(1+s_k/\sigma^2)\approx\log(\sqrt{s_k}/\sigma)$ when
$s_k\gg\sigma^2$. This is the log of the number of noise-width cells that the
signal's spread covers in that direction. The noisy mutual information counts
cells of size $\sigma$, exactly as the binned estimate counts cells of size
$\Delta$. In both, the analyst's choice of scale is what makes the number
finite.

**6.D.ii.1.** $Y$ is a function of $X$, so $I(X;T)=I(X,Y;T)=I(Y;T)+I(X;T\mid Y)$
by the chain rule. Given $Y=y$, $f(X)=ym+W^{1/2}U\sim\mathcal N(ym,W)$, so
by 6.D.i.1, $I(X;T\mid Y=y)=\tfrac12\log\det(I+W/\sigma^2)$ for both values of
$y$.

**6.D.ii.2.** Given $Y=y$, $T\sim\mathcal N(ym,\Sigma_T)$. The log-likelihood
ratio is
$L=\log\frac{p(T\mid+)}{p(T\mid-)}=2m^\top\Sigma_T^{-1}T$. Given $Y=+1$, $L$ has
mean $2m^\top\Sigma_T^{-1}m=2d^2$ and variance
$4m^\top\Sigma_T^{-1}\Sigma_T\Sigma_T^{-1}m=4d^2$. The posterior is
$\Pr[Y=+1\mid T]=1/(1+e^{-L})$, so given $Y=+1$, $-\log\Pr[Y\mid T]=\log(1+e^{-L})$,
and the case $Y=-1$ is the mirror image. Hence
$I(T;Y)=H(Y)-H(Y\mid T)=\log2-\mathbb E\log(1+e^{-L})$, which depends only on
$d^2$. (Since $L$ is a sufficient statistic, $I(T;Y)=I(L;Y)$.)

Monotonicity: $L/(2d)=Yd+G$ with $G\sim\mathcal N(0,1)$. For $d'<d$,
$Yd'+G'\overset{\text{law}}{=}\frac{d'}{d}(Yd+G)+\sqrt{1-d'^2/d^2}\,G''$, so
the $d'$ channel is the $d$ channel followed by independent noise. By data
processing, $I$ is non-decreasing in $d$. As $d\to\infty$, $L\to+\infty$ given
$Y=+1$, so $I\to\log2=H(Y)$.

**6.D.ii.3.** With $W\to\alpha W$:
$I(X;T\mid Y)=\tfrac12\log\det(I+\alpha W/\sigma^2)\downarrow0$, while
$d^2=m^\top(\alpha W+\sigma^2I)^{-1}m\uparrow\|m\|^2/\sigma^2$, so $I(T;Y)$
is nondecreasing. The point moves **left and up** (weakly: a coordinate can
stay fixed in degenerate cases such as $W=0$). To justify the leftward movement, for $\alpha_2>\alpha_1$
the marginal law of $T_{\alpha_2}$ is that of $T_{\alpha_1}$ plus independent
Gaussian noise with covariance $(\alpha_2-\alpha_1)W$. Adding independent
noise cannot decrease differential entropy here, while $h(T\mid X)$ stays
fixed. Thus $I(X;T)$ is nondecreasing in $\alpha$. In the limit $I(X;T)=I(T;Y)$:
the layer carries no information about $X$ beyond what it carries about $Y$. A numerical example from `checks/wb2b_check.py`, with $m=e_1$,
$W=\mathrm{diag}(0.6,1,1)$, $\sigma=0.5$ (nats; Monte Carlo agrees with the
formulas to $\le0.004$):

| $\alpha$ | 1 | 0.3 | 0.1 | 0.01 |
|---|---|---|---|---|
| $I(X;T)$ | 2.59 | 1.59 | 1.04 | 0.68 |
| $I(T;Y)$ | 0.37 | 0.53 | 0.60 | 0.63 |

In this model, compression in the noisy-network sense *is* within-class
clustering. The conditional information
$I(X;T\mid Y)=\tfrac12\log\det(I+W/\sigma^2)$ measures within-class
spread in units of $\sigma$. The drop in total $I(X;T)$ is the drop in
this conditional term **minus** the gain in label information $I(T;Y)$;
it is not equal to the conditional term alone.

**6.D.ii.4.** $f\to cf$ multiplies $W$ by $c^2$ and $m$ by $c$, which is the
same as $\sigma\to\sigma/c$. $I(X;T)$ grows without bound as $c\to\infty$ if $W$ has a positive
eigenvalue. If $W=0$, the signal contains only the binary label and the
information is bounded by $\log2$ (approaching it if $m\ne0$).
The noisy mutual information is therefore not a property of the
representation's shape alone but of its size relative to $\sigma$, the scale
the analyst injected. Read with 6.C, both estimators measure geometry at a
chosen resolution. Whether that makes mutual information the wrong quantity,
or the right one at a stated scale, is not settled by these calculations.
"""
    ),
    markdown(
        r"""
### 6.E. The IB as rate–distortion (optional)

Rate–distortion theory asks for the encoder $p(t\mid x)$ that minimizes the
rate $I(X;T)$ subject to an expected distortion
$\mathbb E\,d(X,T)\le D$ (Cover and Thomas, *Elements of Information Theory*,
ch. 10). Part 3 is the self-consistent equation of Tishby, Pereira and Bialek
(linked in 6.A).

#### 6.E.i. The distortion hidden in the IB

1. Define $p(y\mid t)=\sum_xp(y\mid x)\,p(x\mid t)$ and $d(x,t)=\mathrm{KL}\big(p(y\mid x)\,\|\,p(y\mid t)\big)$. Show that $\mathbb E\,d(X,T)=I(X;Y)-I(T;Y)$, so the IB Lagrangian is the rate–distortion Lagrangian $I(X;T)+\beta\,\mathbb E\,d(X,T)$ up to a constant.
2. Why is this not an ordinary rate–distortion problem?
3. Show that at a stationary point, $p(t\mid x)=\dfrac{p(t)}{Z(x,\beta)}\exp\!\big(-\beta\,\mathrm{KL}(p(y\mid x)\,\|\,p(y\mid t))\big)$.
4. Seen as a regularizer on a network, what does penalizing $I(X;T)$ in the noisy-channel model of 6.D penalize?

<details><summary>Hint</summary>

For 3, differentiate $I(X;T)-\beta I(T;Y)$ with respect to $p(t\mid x)$, with a multiplier $\lambda(x)$ for $\sum_tp(t\mid x)=1$. Terms that depend only on $x$ can be absorbed into $\lambda(x)$.
</details>
"""
    ),
    answer(
        r"""
**1.** With $Y\to X\to T$, the joint is $p(x)p(y\mid x)p(t\mid x)$. Then
$$
\mathbb E\,d=\sum_{x,t,y}p(x,t)\,p(y\mid x)\log\frac{p(y\mid x)}{p(y\mid t)}
=-H(Y\mid X)+H(Y\mid T)=I(X;Y)-I(T;Y),
$$
using $p(x,t)p(y\mid x)=p(x,y,t)$. Hence
$I(X;T)-\beta I(T;Y)=I(X;T)+\beta\,\mathbb E\,d-\beta I(X;Y)$, and the last
term does not depend on the encoder.

**2.** The distortion depends on the encoder itself, through $p(y\mid t)$.
In rate–distortion theory $d$ is fixed in advance. This is why the IB
equations must be solved self-consistently, by iterating.

**3.** $\partial I(X;T)/\partial p(t\mid x)=p(x)\big[\log\frac{p(t\mid x)}{p(t)}\big]+(\text{terms in }x\text{ only})$.
For $I(T;Y)$, with $p(t,y)=\sum_xp(t\mid x)p(x,y)$ and $p(t)=\sum_xp(t\mid x)p(x)$,
$$
\frac{\partial I(T;Y)}{\partial p(t\mid x)}=\sum_yp(x,y)\log\frac{p(y\mid t)}{p(y)}
=p(x)\Big[-\mathrm{KL}\big(p(y\mid x)\|p(y\mid t)\big)+\sum_yp(y\mid x)\log\frac{p(y\mid x)}{p(y)}\Big],
$$
where the terms from differentiating inside the logarithm cancel. The last
sum depends only on $x$. Setting the derivative of the Lagrangian to zero
gives $\log\frac{p(t\mid x)}{p(t)}+\beta\,\mathrm{KL}(\cdot\|\cdot)=-\tilde\lambda(x)$,
which is the stated form with $Z(x,\beta)$ the normalizer.

**4.** By 6.D.i.2, $I(X;T)\le\tfrac12\log\det(I+S/\sigma^2)$, and 6.D.ii.1
splits it into a label term and $\tfrac12\log\det(I+W/\sigma^2)$. Penalizing
it pushes the representation to have small spread in units of $\sigma$, and,
at fixed label information, small within-class spread. It acts as a penalty on
activation scale and on within-class variance: a clustering pressure, once
more a geometric statement.
"""
    ),
    markdown(
        r"""
### 6.F. Readings

"""
        + reading(
            "Opening the black box of deep neural networks via information",
            "https://arxiv.org/abs/1703.00810",
            "Ravid Shwartz-Ziv, Naftali Tishby (arXiv, 2017)",
            "The fitting-then-compression account summarized in 6.A, with its information-plane figures; it "
            "also argues that compression explains the benefit of depth.",
            "Read the IB setup and the description of the binning estimator (30 bins, 4096 inputs), then the "
            "information-plane figures and the drift-then-diffusion account of SGD. Read it against 6.B and "
            "6.C: at each figure, ask what the plotted quantity is for a deterministic network.",
            tier="core",
        )
        + "\n\n"
        + reading(
            "On the information bottleneck theory of deep learning",
            "https://openreview.net/forum?id=ry_WPG-A-",
            "Andrew M. Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, "
            "Brendan D. Tracey, David D. Cox (ICLR 2018)",
            "The critique summarized in 6.A, with experiments comparing tanh, ReLU and linear networks "
            "and full-batch versus stochastic training.",
            "Read the introduction and the sections on the nonlinearity (the binning analysis is the "
            "one you did in 6.C), on compression and generalization, and on batch vs stochastic training. "
            "The linear-network analysis, where task-irrelevant information can be compressed while "
            "fitting, is optional.",
            tier="core",
        )
        + "\n\n"
        + reading(
            "Estimating information flow in deep neural networks",
            "https://arxiv.org/abs/1810.05728",
            "Ziv Goldfeld, Ewout van den Berg, Kristjan Greenewald, Igor Melnyk, Nam Nguyen, "
            "Brian Kingsbury, Yury Polyanskiy (ICML 2019)",
            "The noisy-network reading summarized in 6.A, with a rigorous estimator for I(X;T) in noisy "
            "networks and small worked examples of compression as clustering.",
            "Read the introduction and the setup for why deterministic I(X;T) is degenerate and how noise "
            "fixes it (6.B and 6.D.i), then the section relating compression to clustering, with its small "
            "worked examples. The estimator's convergence theory can be skipped.",
            tier="core",
        )
        + r"""

#### 6.F.i. What the derivations settle

The three accounts in 6.A amount to three positions: (i) compression in mutual
information is real and causes generalization; (ii) it is an artefact of
saturation and binning, not needed for generalization; (iii) it is real
within-class clustering, which binned estimates also track. What does each of
6.B, 6.C and 6.D establish or rule out about these positions? Which question
does none of them address, and what kind of experiment would?
"""
    ),
    answer(
        r"""
- **6.B** rules out the literal reading of (i) for deterministic networks:
  noise-free $I(X;T)$ is constant while the layer stays injective on a
  discrete input set (it can fall only through exact collisions), and
  infinite when the representation is atomless. So "compression in mutual
  information" must mean compression of some estimate, or of a noisy
  surrogate.
- **6.C** supports (ii)'s mechanism. Saturation alone produces rise-then-fall
  in the tanh estimate, and the ReLU estimate grows or stays constant
  depending on the bins. But it also supports (iii): the estimate is a
  spread-at-resolution-$\Delta$ statistic, which is geometric, so "artefact"
  and "measures geometry" are both true.
- **6.D** shows that once noise defines a scale, the drop in $I(X;T)$ at
  fixed label information is exactly within-class shrinkage. That is (iii)'s
  claim, made exact in a toy model. Whether mutual information is the right
  name for it is now a question about invariances (6.D.ii.4). Noise-free
  mutual information is unchanged by any invertible map of $T$; a geometric
  statistic, such as the ratio of within-class to between-class scatter, is
  not. Two representations with equal estimated $I(X;T)$ and $I(T;Y)$ but
  different geometry would separate the two.
- None of 6.B–6.D bears on causation (does clustering *cause*
  generalization?). That needs interventions: for example, penalize any
  decrease in within-class spread and see whether test loss suffers. Two
  observational checks are already in Saxe et al.: ReLU networks generalize
  without compressing in the binned estimate, and the compression phase also
  appears under full-batch training, which rules out SGD noise as its cause.
"""
    ),
    # ------------------------------------------------------------------ §7
    markdown(
        r"""
<a id="wb2-section-7"></a>

## 7. Bayes and MLE generalization error

**The result.** In the regular regime (fixed $p$, $n\to\infty$, a nondegenerate
minimum), the ERM parameter fluctuates with the sandwich covariance
$H^{-1}CH^{-1}/n$. Its expected excess risk is $\mathrm{Tr}(H^{-1}C)/2n$, and
its expected optimism (test minus training loss) is twice that. When the
model is a well-specified likelihood, $C=H$ and these become $p/2n$ and
$p/n$: AIC. The Laplace approximation to the evidence gives BIC's
$\frac p2\log n$. The optional 7.E compares a single fit with the Bayes
ensemble and with a single posterior draw: in the regular case their errors
are $p/2n$, $p/2n$ and $p/n$, and in singular models the real log canonical
threshold replaces $p/2$ in the Bayes error.

*Sources:* the sandwich is due to Huber (1967) and, for misspecified
likelihoods, H. White, "Maximum likelihood estimation of misspecified models",
*Econometrica* (1982). AIC is from Akaike (1974) and BIC from
[Schwarz (1978)](https://doi.org/10.1214/aos/1176344136); ESL §7.4–7.5 and
§7.7 derive both in textbook form.

**Setup, self-contained.** Data $z=(x,y)\sim P$, sample of size $n$, loss
$\ell(\theta;z)$, population risk $R(\theta)=\mathbb E\,\ell(\theta;z)$,
empirical risk $\hat R_n(\theta)=\frac1n\sum_i\ell(\theta;z_i)$, ERM
$\hat\theta_n$, population minimizer $\theta^*$. Assume:

- $p$ is fixed and $n\to\infty$; $\theta^*$ is the unique minimizer, in the interior;
- $H:=\nabla^2R(\theta^*)\succ0$, and $\hat\theta_n\to\theta^*$ in probability;
- enough smoothness and moments for Taylor expansions, the law of large numbers and the central limit theorem.

Define the gradient-noise covariance at the optimum,
$$
C:=\mathrm{Cov}\big[\nabla_\theta\ell(\theta^*;z)\big]=\mathbb E\big[\nabla\ell(\theta^*;z)\nabla\ell(\theta^*;z)^\top\big],
$$
where the two agree because $\mathbb E\nabla\ell(\theta^*;z)=\nabla R(\theta^*)=0$.

From §4, restated: if $\theta=\theta^*+\delta$ with $\mathbb E\,\delta\delta^\top=\Sigma$,
then $\mathbb E\,R(\theta)-R(\theta^*)\approx\tfrac12\mathrm{Tr}(H\Sigma)$,
because $\nabla R(\theta^*)=0$.

**Loss conventions.** All excess risks below are in loss units. For squared
loss this section uses $\ell=\tfrac12(y-f_\theta(x))^2$, so the Gauss–Newton term has no factor of 2; the full Hessian also has
a residual-times-second-derivative term, whose population expectation
vanishes for a correctly specified conditional mean. (§1 used $(y-f)^2$, which doubles $H$
and quadruples $C$; $\mathrm{Tr}(H^{-1}C)$ doubles.) For the log loss
$\ell=-\log p_\theta(y\mid x)$, in nats, and a well-specified model,
$R(\theta)-R(\theta^*)=\mathbb E_x\,\mathrm{KL}\big(p_{\theta^*}(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x)\big)$,
so excess risk is the KL divergence to the truth.

### 7.A. The sandwich

#### 7.A.i. Asymptotic normality of ERM

Starting from the first-order condition $\nabla\hat R_n(\hat\theta_n)=0$, show
that
$$
\sqrt n\,(\hat\theta_n-\theta^*)\ \Rightarrow\ \mathcal N\big(0,\ H^{-1}CH^{-1}\big).
$$
Which of $H$ and $C$ would a "Fisher information" formula use, and when does
that give the wrong answer?

<details><summary>Hint</summary>

Taylor-expand $\nabla\hat R_n$ around $\theta^*$. Apply the law of large numbers to $\nabla^2\hat R_n(\theta^*)$ and the central limit theorem to $\sqrt n\,\nabla\hat R_n(\theta^*)$, then Slutsky.
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

The "inverse Fisher information" formula is $H^{-1}$ (or $C^{-1}$), which
equals the sandwich only when $C=H$ (7.C). Misspecification can break this equality, although it need not do so
in every model. Example: heteroscedastic noise fitted with a homoscedastic
model, as in 7.B.ii.
"""
    ),
    markdown(
        r"""
### 7.B. Excess risk, optimism, and a factor of two

#### 7.B.i. Two traces

1. Show that $\mathbb E\,R(\hat\theta_n)-R(\theta^*)\approx\frac1{2n}\mathrm{Tr}(H^{-1}C)$.
2. Show that the expected optimism is twice as large:
$$
\mathbb E\big[R(\hat\theta_n)-\hat R_n(\hat\theta_n)\big]\approx\frac1n\mathrm{Tr}(H^{-1}C).
$$
This is Takeuchi's information criterion (TIC): estimate the test loss by $\hat R_n(\hat\theta_n)+\frac1n\mathrm{Tr}(\hat H^{-1}\hat C)$.
3. Explain the factor of two in one sentence. Which of the two quantities does a held-out set estimate?

<details><summary>Hint</summary>

For 2, expand $\hat R_n$ around its own minimizer $\hat\theta_n$ and evaluate the expansion at $\theta^*$. Then use $\mathbb E\hat R_n(\theta^*)=R(\theta^*)$.
</details>

#### 7.B.ii. A concrete misspecified case

Let $x\sim\mathcal N(0,1)$ and $y=\beta x+\varepsilon$, where
$\varepsilon\mid x\sim\mathcal N(0,\tau^2x^2)$. The noise is heteroscedastic.
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
**7.B.i.1.** By 7.A, $\delta=\hat\theta_n-\theta^*$ has
$\mathbb E\,\delta\delta^\top\approx\Sigma_n:=H^{-1}CH^{-1}/n$. By the §4
formula, the expected excess risk is
$\tfrac12\mathrm{Tr}(H\Sigma_n)=\frac1{2n}\mathrm{Tr}(HH^{-1}CH^{-1})=\frac1{2n}\mathrm{Tr}(H^{-1}C)$.

**7.B.i.2.** Expand $\hat R_n$ around its minimizer, where its gradient
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

**7.B.i.3.** The test loss of $\hat\theta_n$ is above $R(\theta^*)$ by
$\mathrm{Tr}(H^{-1}C)/2n$, and its training loss is *below* $R(\theta^*)$ by
the same amount, because $\hat\theta_n$ was fitted to that sample's noise. The
optimism spans both. A held-out set estimates $R(\hat\theta_n)$, so held-out
minus training loss estimates the optimism. Held-out loss cannot give the
excess risk without the unknown $R(\theta^*)$, although differences of excess
risk between two models are measurable.

**7.B.ii.1.** $\nabla\ell=-(y-\theta x)x$, so $H=\mathbb E x^2=1$. At
$\theta^*=\beta$, $\nabla\ell=-\varepsilon x$, so
$C=\mathbb E[\varepsilon^2x^2]=\mathbb E[\tau^2x^2\cdot x^2]=3\tau^2$, using
$\mathbb E x^4=3$. Hence:

- $\mathrm{Var}\,\hat\theta_n\approx H^{-1}CH^{-1}/n=3\tau^2/n$;
- excess risk $\approx\mathrm{Tr}(H^{-1}C)/2n=3\tau^2/2n$;
- optimism $\approx3\tau^2/n$.

**7.B.ii.2.** The homoscedastic formulas use $C=\tau^2H$, i.e.
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
### 7.C. The Fisher identity and AIC

#### 7.C.i. When $C=H$

1. Let $\ell(\theta;x,y)=-\log p_\theta(y\mid x)$, with the data generated by $p_{\theta^*}$ (a well-specified model). Prove that $C=H$ at $\theta^*$ by differentiating $\int p_\theta(y\mid x)\,dy=1$ twice.
2. Conclude that the optimism is $p/n$ and the excess risk $p/2n$. Relate this to AIC $=-2\log L(\hat\theta)+2p$.
3. For squared loss $\ell=\tfrac12(y-f_\theta(x))^2$ with $y=f_{\theta^*}(x)+\varepsilon$, $\varepsilon$ independent of $x$ and of variance $\sigma^2$, show that $C=\sigma^2H$, so $\mathrm{Tr}(H^{-1}C)=\sigma^2p$. Check this against OLS from §1, minding the convention.
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
in the $\tfrac12$ convention, i.e. $\sigma^2p/n$ in §1's $(y-f)^2$
convention. That matches the large-$n$ limit of §1's exact
$\sigma^2p/(n-p-1)$.
"""
    ),
    markdown(
        r"""
### 7.D. The Bayesian side: Laplace and BIC

#### 7.D.i. The evidence to leading order

Take the log loss, a prior density $\pi$ that is continuous and positive at
$\theta^*$, and the marginal likelihood (evidence)
$$
Z_n=\int\pi(\theta)\,e^{-n\hat R_n(\theta)}\,d\theta .
$$

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
with $\lambda$ the real log canonical threshold, the central quantity of singular learning theory (SLT).
"""
    ),
    markdown(
        r"""
### 7.E. Three estimands (optional)

Use the log loss, a model $p_\theta(y\mid x)$, and a truth $q(y\mid x)$; take
the model to be well specified, $q=p_{\theta^*}$. Let
$\pi_n(\theta)\propto\pi(\theta)\prod_ip_\theta(y_i\mid x_i)$ be the
posterior, and $\bar p_n(y\mid x)=\int p_\theta(y\mid x)\,\pi_n(d\theta)$ the
posterior predictive. With $\mathbb E_D$ over the training sample:

1. **Single-network excess risk:** $\ \mathcal E=\mathbb E_D\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,p_{\hat\theta_n}\big)=\mathbb E_D R(\hat\theta_n)-R(\theta^*)$. One deterministic output of training.
2. **Bayes generalization error:** $\ B_g=\mathbb E_D\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,\bar p_n\big)$. The posterior-predictive ensemble.
3. **Gibbs generalization error:** $\ G_g=\mathbb E_D\,\mathbb E_{\theta\sim\pi_n}\,\mathbb E_x\,\mathrm{KL}\big(q\,\|\,p_\theta\big)$. One random draw from the posterior.

Each is an expected population loss minus the entropy $R(\theta^*)$ of the
truth, so all three are in nats per sample. (Watanabe, *Mathematical Theory
of Bayesian Statistics*, 2018, ch. 1–2, defines them without the outer
$\mathbb E_D$, states results for their expectations, and treats the singular
case. For the singular Bayes generalization error and WAIC, see also
[Watanabe (2010)](https://jmlr.org/papers/v11/watanabe10a.html).)

**Watanabe's regular-case results.** For a regular, well-specified model at
temperature 1, $\mathbb E\,B_g=\frac p{2n}+o(\frac1n)$ and
$\mathbb E\,G_g=\frac pn+o(\frac1n)$. Under the realizable analytic assumptions of singular learning theory,
$\mathbb E\,B_g=\frac\lambda n+o(\frac1n)$: the real log canonical threshold
$\lambda\le p/2$ replaces $p/2$.

#### 7.E.i. Deriving the regular case

1. Show that $B_g\le G_g$ always.
2. In the regular case, approximate the posterior by $\mathcal N\big(\hat\theta_n,(nH)^{-1}\big)$. Show that $G_g\approx p/n$.
3. Show that $B_g\approx p/2n$, the same as $\mathcal E$ to leading order.
4. Without well-specification, define $K^*=\mathbb E_x\mathrm{KL}(q\|p_{\theta^*})$. Keep the same Laplace posterior and compute $G_g-K^*$ in terms of $\mathrm{Tr}(H^{-1}C)$. What does this say about the posterior's width?

<details><summary>Hint</summary>

For 1, $\mathrm{KL}(q\|\cdot)$ is convex in its second argument. For 2, write a posterior draw as $\theta=\hat\theta_n+u$, with $\mathbb E[u\mid D]=0$ and $\delta=\hat\theta_n-\theta^*$ fixed given the data. For 3, expand $\log\mathbb E_u\,p_{\hat\theta_n+u}(y\mid x)$ to second order in $u$, and use the Fisher identity of 7.C.
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
#### 7.E.ii. Which estimand do practical tools measure?

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
  for variance (§3–§4). Ridge is MAP under a Gaussian prior, which is neither
  the Bayes ensemble nor a Gibbs draw. (In linear-Gaussian models the
  posterior mean equals the MAP, so there it coincides with the ensemble's
  mean prediction.)
- **Flatness** $\tfrac12\mathrm{Tr}(H\Sigma)$ approximates the mean local
  excess risk when $\Sigma=\mathbb E[(\theta-\theta^*)(\theta-\theta^*)^\top]$.
  This is a second moment, equal to the covariance only for a centered parameter. If $\Sigma$ is the sampling covariance of
  $\hat\theta_n$, it is estimand 1 (7.B). If $\Sigma$ is the posterior's
  covariance, it is the posterior's contribution to estimand 3. If $\Sigma$ is
  SGD's stationary covariance, it is a Gibbs-like error for the "SGD
  posterior" (§8). Which one is meant has to be stated.
- None of the regular-case formulas is a theorem for overparametrized
  networks trained by SGD: $H$ can be singular, and the empirical gradient
  covariance can vanish at squared-loss interpolation even when the
  population covariance does not (§8).
"""
    ),
    # ------------------------------------------------------------------ §8
    markdown(
        r"""
<a id="wb2-section-8"></a>

## 8. Does it cancel? $C$ vs $H$

**The result.** The clean formulas of §7 lean on $C\propto H$, which holds for
a well-specified likelihood at the optimum. Per-sample, both matrices are
built from the same rank-one pieces, but weighted differently, and at
squared-loss interpolation the empirical gradient covariance vanishes
while empirical curvature can remain nonzero. Empirically the diagonals follow
$C_{ii}\propto H_{ii}^\gamma$, with fitted $\gamma$ roughly between 1 and 1.5,
and a moment argument (8.B) makes $[1,2]$ the natural range, though it does
not force a fitted exponent into it. On a quadratic basin, in
the continuous-time approximation, SGD's stationary covariance solves
$H\Sigma+\Sigma H=(\eta/B)C$. It matches a
tempered posterior $\frac TnH^{-1}$ if and only if $C\propto I$, i.e.
$\gamma=0$, which is not the measured range. In the continuous-time approximation its quadratic excess risk is
$\eta\,\mathrm{Tr}\,C/4B$ exactly; discrete SGD has a step-size correction.

**Setup, self-contained.** Loss $\ell(\theta;z)$, population risk $R$,
minimizer $\theta^*$ (shifted to $0$ where convenient),
$H=\nabla^2R(\theta^*)$, and gradient-noise covariance
$C=\mathrm{Cov}[\nabla\ell(\theta;z)]$. From §4: a parameter
$\theta^*+\delta$ with $\mathbb E\,\delta\delta^\top=\Sigma$ has excess risk
$\approx\tfrac12\mathrm{Tr}(H\Sigma)$. From §7: for a well-specified log loss,
$C=H$ at $\theta^*$, and for squared loss with independent noise, $C=\sigma^2H$.
A *tempered posterior* is $\pi_T(\theta)\propto\pi(\theta)\,e^{-n\hat R_n(\theta)/T}$;
near a nondegenerate mode it has covariance $\approx\frac TnH^{-1}$.

### 8.A. Per-sample structure

*Source:* Zhang, Yang and Tu (reading in 8.D), §2.1, on why the Fisher
approximation $C\approx H$ fails.

#### 8.A.i. Same pieces, different weights

1. *Squared loss.* Let $\ell_i=\tfrac12r_i^2$ with residual $r_i=f_\theta(x_i)-y_i$, and $v_i=\nabla f_\theta(x_i)$. Write the per-sample gradient $g_i$ and the per-sample Gauss–Newton matrix $h_i$ (the Hessian without the $r_i\nabla^2f$ term). Show that $h_i$ has rank at most one, and that over the training set
$$
H_{GN}=\frac1n\sum_ih_i,\qquad C=\frac1n\sum_ir_i^2h_i-\bar g\bar g^\top .
$$
2. At an interpolating minimum ($r_i=0$ for all $i$), what are $C$ and $H_{GN}$? Away from interpolation, under what condition is $C=\sigma^2H_{GN}$?
3. *Logistic loss.* With $y_i\in\{0,1\}$, $s_i=\sigma(f_\theta(x_i))$ and $\ell_i=-y_i\log s_i-(1-y_i)\log(1-s_i)$, show that $g_i=(s_i-y_i)v_i$ and that the per-sample Gauss–Newton matrix is $h_i=a_iv_iv_i^\top$ with $a_i=s_i(1-s_i)$. For a correctly classified example with large margin, compare the weights of $v_iv_i^\top$ in $g_ig_i^\top$ and in $h_i$. Which goes to zero faster as the margin grows?
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
### 8.B. The diagonal power law

Zhang, Yang and Tu (2026) argue that, beyond the Fisher-equals-Hessian
setting, $C\propto\mathbb E_i[h_i^2]$ (with $h_i$ the per-sample Hessian), and also report approximate commutation of
$C$ and $H=\mathbb E_i[h_i]$ in their setting. Commutation is an additional
property, not a general consequence of the second-moment relation. In the
fully connected layers they analyse, they fit per-layer power laws
$C_{ii}\propto H_{ii}^\gamma$, with fitted $\gamma$ roughly $1.1$–$1.5$ for
cross-entropy and close to $1$ for squared loss. Their §5 derives a pair of
moment bounds and reads them as confining $\gamma$ to $[1,2]$. This exercise
derives the same kind of bounds in a simplified model and asks what they do
and do not imply.

#### 8.B.i. Why between 1 and 2?

Take rank-one per-sample Hessians $h_i=v_iv_i^\top$ as in 8.A.i.1, and the
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
### 8.C. SGD's stationary law on a quadratic basin

Near the minimum (shifted to $0$), take $R(\theta)\approx\tfrac12\theta^\top H\theta$
with $H\succ0$, and SGD with learning rate $\eta$ and batch size $B$:
$$
\theta_{k+1}=\theta_k-\eta\,(H\theta_k+\xi_k),\qquad\mathbb E\,\xi_k=0,\quad\mathrm{Cov}\,\xi_k=C/B,
$$
with $\xi_k$ independent across steps and independent of $\theta_k$, and
$C$ fixed (state-independent near the minimum). For discrete-time stationarity
assume $0<\eta<2/\lambda_{\max}(H)$.

*Source:* this stationary-distribution analysis, and the idea of tuning
constant-step SGD so that its stationary law approximates a posterior, are in
[Mandt, Hoffman and Blei (2017)](https://arxiv.org/abs/1704.04289).

#### 8.C.i. The Lyapunov equation and the posterior

1. Show that the stationary covariance of the recursion satisfies $\Sigma_\eta=(I-\eta H)\Sigma_\eta(I-\eta H)+\eta^2C/B$, and that to leading order in $\eta$ this is the Lyapunov equation
$$
H\Sigma+\Sigma H=\frac\eta B\,C .
$$
(It is also the stationary equation of the Ornstein–Uhlenbeck process $d\theta=-H\theta\,dt+\sqrt{\eta/B}\,C^{1/2}dW$, with $t=\eta k$.)
2. Show that the Lyapunov equation has a unique solution for $H\succ0$. If $C$ and $H$ commute, show that $\Sigma=\frac\eta{2B}CH^{-1}$. With $C=cH^\gamma$, this is $\Sigma=\frac{\eta c}{2B}H^{\gamma-1}$.
3. Show that $\Sigma=\frac TnH^{-1}$ for some temperature $T$ if and only if $C\propto I$. Do not assume that $C$ and $H$ commute. Find $T$.
4. Show that $\tfrac12\mathrm{Tr}(H\Sigma)=\frac{\eta\,\mathrm{Tr}\,C}{4B}$ exactly, for any $C$.
5. With $C=cH^\gamma$, write three excess risks: ERM's $\frac1{2n}\mathrm{Tr}(H^{-1}C)$ from §7, SGD's stationary value, and the tempered posterior's $\tfrac12\mathrm{Tr}(H\cdot\frac TnH^{-1})$. What does each count?
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
Lyapunov equation. (The discrete law is wider in sharp directions, by a
relative $O(\eta h)$, and diverges at $\eta h\ge2$.)

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
| ERM (§7) | $\frac c{2n}\mathrm{Tr}\,H^{\gamma-1}$ | sampling noise in $\hat\theta_n$, each direction weighted by noise over curvature |
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
#### 8.C.ii. The best temperature, and a symmetry (optional)

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
(`checks/wb2b_check.py`). The interactive figure after 8.C.iii uses $p=2$, so the
symmetry of its $\mathrm{KL}^*$ readout is a feature of $p=2$.
"""
    ),
    markdown(
        r"""
#### 8.C.iii. A two-dimensional example

Take $H=\mathrm{diag}(1,10)$: $\theta_1$ is flat and $\theta_2$ sharp. With
$C=cH^\gamma$, use 8.C.i.2 to find the axes and aspect ratio of SGD's
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
  $\Sigma\propto H^{-1}$ requires $C\propto I$ (8.C.i.3), no temperature
  fixes the shape, and $\mathrm{KL}^*>0$ for every $T$.
- The discrete-time ellipse is slightly wider than the continuous one along
  $\theta_2$, by a relative $O(\eta\kappa)$.

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
### 8.D. Readings and rabbit holes

"""
        + reading(
            "On the superlinear relationship between SGD noise covariance and loss landscape curvature",
            "https://arxiv.org/abs/2602.05600",
            "Yikuan Zhang, Ning Yang, Yuhai Tu (arXiv, 2026)",
            "The assumption that SGD noise covariance is proportional to the Hessian holds only under "
            "restrictive conditions that deep networks typically violate. Using an activity–weight duality, "
            "they find C ∝ E_p[h_p²] instead. They also report that C and H approximately commute without "
            "coinciding (a separate property; see 8.B), and that the diagonals follow per-layer power laws C_ii ∝ H_ii^γ, with fitted γ near 1–1.5 "
            "and an argument that 1 ≤ γ ≤ 2.",
            "Read §2 (why the Fisher-equals-Hessian argument fails: compare 8.A) and skim §3, whose "
            "§3.4 derives C ∝ E[h_p²]. Then read §4 (the measured power laws, on MLPs and small CNNs on "
            "MNIST and CIFAR-10) and §5 (the moment bounds on γ, which are the envelope of 8.B.i, and why "
            "cross-entropy gives larger γ than squared loss). Note which layers the fits cover before "
            "carrying γ into 8.C.",
            tier="core",
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
            "It is a statement about Bayesian learning (§7.E), not about SGD. The comment thread "
            "discusses exactly that gap (some circuit-learnable functions, like sparse parities, are hard "
            "for gradient descent). Read it with 8.C.i.7 in mind.",
            tier="rabbit hole",
        )
    ),
    # ------------------------------------------------------------------ §9
    markdown(
        r"""
<a id="wb2-section-9"></a>

## 9. Fantastic generalization measures

This section is a reading, with one exercise that connects it to §4, §7 and
§8.

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
            tier="core",
        )
        + r"""

### 9.A. Exercise: invariance, and the assumptions behind TIC

1. Let $\theta=A\phi$ with $A$ invertible be a reparametrization that leaves the network function fixed (or the linearization of a smooth one at the minimum). Show that at the minimum $H\to A^\top HA$ and $C\to A^\top CA$, and that $\mathrm{Tr}(H^{-1}C)$ is invariant. Which of the parameter $\ell_2$ norm, the gradient variance $\mathrm{Tr}\,C$, $\mathrm{Tr}\,H$ and the product of layer spectral norms are invariant under the layer rescaling of §4.E.i.3, and which under a general $A$?
2. Jiang et al. train every model to the same cross-entropy. Why? Using 8.A, what does this do to the per-sample gradients and to $\hat C$? Which assumptions behind the derivation of TIC in 7.B fail for these networks?
3. Jiang et al. report both the pooled Kendall $\tau$ and the granulated $\Psi$ (their §2.2). Why might a measure have a high pooled $\tau$ but a low $\Psi$?

<details><summary>Hint</summary>

For 1, per-sample gradients transform as $\nabla_\phi\ell=A^\top\nabla_\theta\ell$, and at a critical point the Hessian transforms as in 4.E.i.4. For 2, list the assumptions in §7's setup (fixed $p$ with $n\to\infty$, $H\succ0$, the trained parameter is the ERM) and check each.
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
- *Invariant to the layer rescaling of §4 but not to general
  reparametrizations:* the product of layer spectral norms, since
  $\|\alpha W_1\|\,\|W_2/\alpha\|=\|W_1\|\,\|W_2\|$ in a ReLU network.
- *Functions of the function:* the margin, and the gap itself.

**2.** Fixing the training loss removes a trivial predictor: models with
higher training loss have a different gap for uninteresting reasons. The
price is that every model is near interpolation, where per-sample gradients
are small (8.A). $\hat C$ is then small and dominated by the hardest
examples, and the regular-case theory behind $\mathrm{Tr}(H^{-1}C)$ is
furthest from applying. The assumptions that fail:
- *Interpolation* (8.A): at exact squared-loss interpolation, the empirical
  $\hat C$ is zero even though population gradient variability and test
  loss can remain positive. Plug-in TIC then loses its regular-regime
  justification. Jiang et al. stop at a small positive cross-entropy,
  not exact interpolation; for cross-entropy both curvature and gradient
  covariance can shrink, at different rates.
- *Singular $H$*: overparametrized networks have flat directions, so
  $H^{-1}$ is undefined and damped versions depend on $\epsilon$. The
  regular-case $p/2$ becomes the RLCT (§7.D–7.E), which TIC does not see.
- *SGD is not ERM*: the trained parameter's spread is the stationary law of
  §8.C, set by $\eta/B$ and $\mathrm{Tr}\,C$, not the sandwich. Learning-rate
  and batch-size axes move the gap through a channel TIC does not model.
  ($\mathrm{Tr}\,C$ itself does model it, which is one reading of Jiang et
  al.'s gradient-variance result.)
- *Wrong loss*: TIC is about the training loss. The 0–1 gap can move while
  the cross-entropy optimism does not, and conversely.
- *Ensembles*: TIC is about a single fitted network. If what you report is
  an ensemble's error, the leading constant can change (§7.E).

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
