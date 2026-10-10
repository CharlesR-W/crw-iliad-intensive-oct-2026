"""WB2 §1.F: information bottleneck. IB_OPENER (pitch, primer, 1.F.1 rate–distortion) is in
wb2_new_ratedistortion.py; IB_RD is 1.F.2; IB_PLANE is the information-plane debate (1.F.3–1.F.7),
checked by checks/wb2b_check.py.
"""
from nbkit import *  # noqa: F401,F403
from wb2_new_ratedistortion import CELLS as IB_OPENER  # noqa: F401  (pitch, primer, 1.F.1)

IB_PLANE = [
    # ------------------------------------------------------------------ §1.F
    markdown(
        r"""
<a id="wb2-section-1f-plane"></a>

#### The information plane: did deep networks compress?

**Why this is here.** Do trained networks actually compress in the IB sense?
Shwartz-Ziv and Tishby (2017) said yes, and that this is why they generalize.
It was one of the most discussed theories of deep learning, and the debate
that followed is a compact lesson in what mutual information can and cannot
measure about a network. The exercises 1.F.3–1.F.6 derive the three facts the
debate turned on; 1.F.7 has the papers. Entropy, mutual information and the
data-processing inequality are defined in the primer at the start of 1.F.

**Setting.** $X$ is the network input, $Y$ the label, and $T=f(X)$ the
activations of one hidden layer, a deterministic function of $X$; $T$ is random
only because $X$ is. Because $T$ depends on $Y$ only through $X$, the variables
form a Markov chain $Y\to X\to T$, so $I(T;Y)\le I(X;Y)$. The *information
plane* plots each layer as the point $\big(I(X;T),\,I(T;Y)\big)$ and follows
it through training. In IB terms (1.F.2), moving left means compressing and
moving up means keeping label information.

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
problem, which 1.F.3 makes precise: for a deterministic network the quantity
being plotted is constant or infinite.

**A third reading** (Goldfeld et al. 2019). Add a
little Gaussian noise to each layer, so that $I(X;T)$ is finite and depends on
the weights. Then compression does occur, and it is driven by the clustering
of same-class representations. The binned estimates track the same clustering.

**What the exercises show.** For a deterministic network, $I(X;T)$ is fixed
by the input distribution as long as the layer is injective on a discrete
input set, and infinite if the activations are continuously distributed
(1.F.3), so it cannot "fall". What a binned estimate measures instead is the
entropy of a quantization of $T$: saturating $\tanh$ units make it rise and
then fall, while for ReLU units it grows or stays constant depending on how
the bins are chosen (1.F.4). Once Gaussian noise is added to the layer,
$I(X;T)$ is finite and counts distinguishable cells at the noise scale (1.F.5).
In a two-class model, at fixed label information $I(T;Y)$, it falls exactly
when the within-class spread, measured in units of the noise, shrinks
(1.F.6).
"""
    ),
    markdown(
        r"""
#### 1.F.3. A deterministic network has nothing to compress

**Why.** The information-plane plots show $I(X;T)$ falling during training.
Before asking why, ask what $I(X;T)$ is when $T$ is a deterministic function of
$X$, as it is in an ordinary network.

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

**2.** $H(X)=\log4096=12$ bits $\approx8.32$ nats, for every such layer, at every step of training, as long as the layer stays injective on the 4096 inputs. Exact collisions are possible but need a mechanism: units that are exactly zero on whole regions of input (dead ReLUs), or saturated $\tanh$ units that round to exactly $\pm1$ in floating point, can map distinct inputs to the same vector. Without such collisions, nothing in $I(X;T)$ changes, whatever a binned estimate does.

**3.** Let $q_\Delta$ map $T$ to the cube of side $\Delta$ that contains it. $q_\Delta(T)$ is a deterministic function of $X$, so by 1, $I(X;q_\Delta(T))=H(q_\Delta(T))$. Since $X\to T\to q_\Delta(T)$, data processing gives $I(X;T)\ge H(q_\Delta(T))\ge-\log\max_k\Pr[q_\Delta(T)=k]$.

It remains to show that $m(\Delta):=\sup_{\text{cubes }Q\text{ of side }\Delta}\Pr[T\in Q]\to0$. Suppose not: then there are $\epsilon>0$ and cubes $Q_j$ with sides $\Delta_j\to0$ and $\Pr[T\in Q_j]\ge\epsilon$. By tightness, the $Q_j$ eventually lie in a compact set, so a subsequence of their centres converges to some $t_0$. Every ball around $t_0$ then has probability at least $\epsilon$, so $\Pr[T=t_0]\ge\epsilon$, contradicting atomlessness. Hence $H(q_\Delta(T))\to\infty$ and $I(X;T)=\infty$.

(Equivalently, when $T$ has a density, $h(T\mid X)=-\infty$ because $T$ given $X$ is a point mass.)

**4.** In case 3, none: $I(X;T)=\infty$ for every weight setting that keeps $T$ atomless. In case 1, only exact collisions, where two distinct inputs are mapped to exactly the same activation vector, can lower $H(f(X))$. Such collisions are possible, for example with dead ReLUs. The constancy conclusion requires injectivity on the discrete input support; the infinite-information conclusion requires atomlessness. Under those conditions, movement in a finite plotted estimate does not describe a change in the exact mutual information.
"""
    ),
    markdown(
        r"""
#### 1.F.4. What the binned estimator measures: saturation looks like compression

**Why.** If the true $I(X;T)$ cannot fall, something else produced the
published curves. The estimator is the place to look.

The binned estimator replaces $T$ by $\hat T=\mathrm{bin}(T)$, its bin index,
and reports $I(X;\hat T)$. Since $\hat T$ is a function of $X$, 1.F.3.1 says
this is just $H(\mathrm{bin}(T))$, the entropy of a quantization.


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
#### 1.F.5. With noise, mutual information is finite: the Gaussian channel

**Why.** 1.F.3 showed that noise-free $I(X;T)$ is degenerate. Goldfeld et al.
make it meaningful by adding noise to the layer:
$T=f(X)+\sigma\varepsilon$ with $\varepsilon\sim\mathcal N(0,I_d)$ independent
of $X$. Now $h(T\mid X)=h(\sigma\varepsilon)=\frac d2\log(2\pi e\sigma^2)$ is
finite, and $I(X;T)=h(T)-\frac d2\log(2\pi e\sigma^2)$ depends on $f$.
Differential entropy and the Gaussian channel are in Cover and Thomas,
*Elements of Information Theory*, ch. 8–9.

1. Suppose $f(X)\sim\mathcal N(\mu,S)$. Show that $I(X;T)=\tfrac12\log\det(I+S/\sigma^2)=\sum_k\tfrac12\log(1+s_k/\sigma^2)$, where $s_k$ are the eigenvalues of $S$.
2. Show that for any $f(X)$ with covariance $S$, $I(X;T)\le\tfrac12\log\det(I+S/\sigma^2)$.
3. Interpret $\tfrac12\log(1+s_k/\sigma^2)$ in terms of the number of distinguishable cells of size $\sigma$ along the $k$-th principal direction. Compare with 1.F.4.5.

<details><summary>Hint</summary>

$h(T\mid X)=h(\sigma\varepsilon)$ is the same for every $f$. For 2, among all densities with a given covariance, the Gaussian has the largest differential entropy.
</details>
"""
    ),
    answer(
        r"""
**1.** $T\sim\mathcal N(\mu,S+\sigma^2I)$, so
$h(T)=\tfrac12\log\det\big(2\pi e(S+\sigma^2I)\big)$ and
$h(T\mid X)=\tfrac12\log\det(2\pi e\sigma^2I)$. Subtracting gives
$\tfrac12\log\det(I+S/\sigma^2)$. Diagonalizing $S$ gives the sum.

**2.** $h(T\mid X)$ is the same for every $f$. $T$ has covariance
$S+\sigma^2I$, and the Gaussian maximizes differential entropy at fixed
covariance, so $h(T)\le\tfrac12\log\det(2\pi e(S+\sigma^2I))$.

**3.** $\tfrac12\log(1+s_k/\sigma^2)\approx\log(\sqrt{s_k}/\sigma)$ when
$s_k\gg\sigma^2$. This is the log of the number of noise-width cells that the
signal's spread covers in that direction. The noisy mutual information counts
cells of size $\sigma$, exactly as the binned estimate counts cells of size
$\Delta$. In both, the analyst's choice of scale is what makes the number
finite.
"""
    ),
    markdown(
        r"""
#### 1.F.6. With noise, compression is clustering: a two-class model

**Why.** Once $I(X;T)$ is finite, what does a decrease in it mean
geometrically? In the simplest model with a label, it means that same-class
representations cluster, and the IB objective of 1.F.2 becomes an explicit
penalty on within-class spread.

Let $Y\in\{\pm1\}$ be uniform, and let $X=(Y,U)$ with $U\sim\mathcal N(0,I_d)$
independent of $Y$, so the label is a function of the input. The layer computes
$$
f(X)=Y\,m+W^{1/2}U ,
$$
so $m\in\mathbb R^d$ sets the class means $\pm m$ (between-class covariance
$mm^\top$) and $W$ is the within-class covariance. Then $T=f(X)+\sigma\varepsilon$.

1. Show that $I(X;T)=I(T;Y)+I(X;T\mid Y)$, and that $I(X;T\mid Y)=\tfrac12\log\det(I+W/\sigma^2)$.
2. Let $\Sigma_T=W+\sigma^2I$ and $\rho^2=m^\top\Sigma_T^{-1}m$ (the squared separation of the class means, measured in the metric of the within-class-plus-noise covariance). Show that $I(T;Y)$ depends on the model only through $\rho^2$, and that
$$
I(T;Y)=\log2-\mathbb E\log\!\big(1+e^{-L}\big),\qquad L\sim\mathcal N(2\rho^2,\,4\rho^2).
$$
Show that this increases with $\rho^2$ and tends to $\log2$.
3. Now shrink the within-class covariance, $W\to\alpha W$ with $\alpha\downarrow0$, keeping $m$ and $\sigma$ fixed. Describe the path of the point $\big(I(X;T),I(T;Y)\big)$. What are its limits?
4. What happens to $I(X;T)$ if you rescale the whole representation, $f\to cf$, and leave $\sigma$ fixed? What does this say about the role of $\sigma$?
5. *Back to the IB.* Write the IB objective of 1.F.2, $I(X;T)-\beta I(T;Y)$ with $\beta>1$, for this model in terms of $W$ and $I(T;Y)$. Seen as a regularizer on the layer, what does it penalize and what does it reward?

<details><summary>Hint</summary>

For 1, use the chain rule on $I(X,Y;T)$ and the fact that $Y$ is a function of $X$. Given $Y$, $f(X)$ is Gaussian, so 1.F.5.1 applies. For 2, the posterior log-odds $L=\log\frac{p(T\mid Y=+1)}{p(T\mid Y=-1)}$ is linear in $T$. For monotonicity, show that a channel with smaller $\rho^2$ is the channel with larger $\rho^2$ followed by extra noise.
</details>
"""
    ),
    answer(
        r"""
**1.** $Y$ is a function of $X$, so $I(X;T)=I(X,Y;T)=I(Y;T)+I(X;T\mid Y)$
by the chain rule. Given $Y=y$, $f(X)=ym+W^{1/2}U\sim\mathcal N(ym,W)$, so
by 1.F.5.1, $I(X;T\mid Y=y)=\tfrac12\log\det(I+W/\sigma^2)$ for both values of
$y$.

**2.** Given $Y=y$, $T\sim\mathcal N(ym,\Sigma_T)$. The log-likelihood
ratio is
$L=\log\frac{p(T\mid+)}{p(T\mid-)}=2m^\top\Sigma_T^{-1}T$. Given $Y=+1$, $L$ has
mean $2m^\top\Sigma_T^{-1}m=2\rho^2$ and variance
$4m^\top\Sigma_T^{-1}\Sigma_T\Sigma_T^{-1}m=4\rho^2$. The posterior is
$\Pr[Y=+1\mid T]=1/(1+e^{-L})$, so given $Y=+1$, $-\log\Pr[Y\mid T]=\log(1+e^{-L})$,
and the case $Y=-1$ is the mirror image. Hence
$I(T;Y)=H(Y)-H(Y\mid T)=\log2-\mathbb E\log(1+e^{-L})$, which depends only on
$\rho^2$. (Since $L$ is a sufficient statistic, $I(T;Y)=I(L;Y)$.)

Monotonicity: $L/(2\rho)=Y\rho+G$ with $G\sim\mathcal N(0,1)$. For $\rho'<\rho$,
$Y\rho'+G'\overset{\text{law}}{=}\frac{\rho'}{\rho}(Y\rho+G)+\sqrt{1-\rho'^2/\rho^2}\,G''$, so
the $\rho'$ channel is the $\rho$ channel followed by independent noise. By data
processing, $I$ is non-decreasing in $\rho$. As $\rho\to\infty$, $L\to+\infty$ given
$Y=+1$, so $I\to\log2=H(Y)$.

**3.** With $W\to\alpha W$:
$I(X;T\mid Y)=\tfrac12\log\det(I+\alpha W/\sigma^2)\downarrow0$, while
$\rho^2=m^\top(\alpha W+\sigma^2I)^{-1}m\uparrow\|m\|^2/\sigma^2$, so $I(T;Y)$
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

**4.** $f\to cf$ multiplies $W$ by $c^2$ and $m$ by $c$, which is the
same as $\sigma\to\sigma/c$. $I(X;T)$ grows without bound as $c\to\infty$ if $W$ has a positive
eigenvalue. If $W=0$, the signal contains only the binary label and the
information is bounded by $\log2$ (approaching it if $m\ne0$).
The noisy mutual information is therefore not a property of the
representation's shape alone but of its size relative to $\sigma$, the scale
the analyst injected. Read with 1.F.4, both estimators measure geometry at a
chosen resolution. Whether that makes mutual information the wrong quantity,
or the right one at a stated scale, is not settled by these calculations.

**5.** By part 1, $I(X;T)=I(T;Y)+\tfrac12\log\det(I+W/\sigma^2)$, so
$$
I(X;T)-\beta I(T;Y)=\tfrac12\log\det(I+W/\sigma^2)-(\beta-1)\,I(T;Y).
$$
The IB objective penalizes within-class spread in units of $\sigma$ and
rewards label information, which by part 2 means a large separation
$\rho^2=m^\top(W+\sigma^2I)^{-1}m$ between the class means. Minimizing it pushes
toward tight, well-separated class clusters: a clustering pressure, once more a
geometric statement whose scale is set by $\sigma$ (part 4). (For $\beta\le1$
the objective is minimized by $W=0$, $m=0$: the trivial representation, as at
low $\beta$ in 1.F.1.7.)
"""
    ),
    markdown(
        r"""
#### 1.F.7. Readings, and what the derivations settle

"""
        + reading(
            "Opening the black box of deep neural networks via information",
            "https://arxiv.org/abs/1703.00810",
            "Ravid Shwartz-Ziv, Naftali Tishby (arXiv, 2017)",
            "The fitting-then-compression account summarized above, with its information-plane figures; it "
            "also argues that compression explains the benefit of depth.",
            "Read the IB setup and the description of the binning estimator (30 bins, 4096 inputs), then the "
            "information-plane figures and the drift-then-diffusion account of SGD. Read it against 1.F.3 and "
            "1.F.4: at each figure, ask what the plotted quantity is for a deterministic network.",
            importance=4, length=2,
        )
        + "\n\n"
        + reading(
            "On the information bottleneck theory of deep learning",
            "https://openreview.net/forum?id=ry_WPG-A-",
            "Andrew M. Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, "
            "Brendan D. Tracey, David D. Cox (ICLR 2018)",
            "The critique summarized above, with experiments comparing tanh, ReLU and linear networks "
            "and full-batch versus stochastic training.",
            "Read the introduction and the sections on the nonlinearity (the binning analysis is the "
            "one you did in 1.F.4), on compression and generalization, and on batch vs stochastic training. "
            "The linear-network analysis, where task-irrelevant information can be compressed while "
            "fitting, is optional.",
            importance=4, length=3,
        )
        + "\n\n"
        + reading(
            "Estimating information flow in deep neural networks",
            "https://arxiv.org/abs/1810.05728",
            "Ziv Goldfeld, Ewout van den Berg, Kristjan Greenewald, Igor Melnyk, Nam Nguyen, "
            "Brian Kingsbury, Yury Polyanskiy (ICML 2019)",
            "The noisy-network reading summarized above, with a rigorous estimator for I(X;T) in noisy "
            "networks and small worked examples of compression as clustering.",
            "Read the introduction and the setup for why deterministic I(X;T) is degenerate and how noise "
            "fixes it (1.F.3 and 1.F.5), then the section relating compression to clustering, with its small "
            "worked examples. The estimator's convergence theory can be skipped.",
            importance=3, length=2,
        )
        + "\n\n"
        + reading(
            "Deep variational information bottleneck",
            "https://arxiv.org/abs/1612.00410",
            "Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy (ICLR 2017)",
            "Turns the IB objective of 1.F.2 into a trainable loss for a network with a stochastic "
            "(Gaussian) encoder, using variational bounds on I(X;T) and I(T;Y), and reports better "
            "generalization and adversarial robustness than deterministic networks with other regularizers.",
            "Read §3 for the bounds and §4.1 for the MNIST results; the encoder's noise plays the role "
            "of σ in 1.F.5–1.F.6.",
            importance=2, length=2,
        )
        + r"""

**Question.** The three accounts of the information plane amount to three positions: (i) compression in mutual
information is real and causes generalization; (ii) it is an artefact of
saturation and binning, not needed for generalization; (iii) it is real
within-class clustering, which binned estimates also track. What does each of
1.F.3, 1.F.4 and 1.F.5–1.F.6 establish or rule out about these positions? Which
question does none of them address, and what kind of experiment would?
"""
    ),
    answer(
        r"""
- **1.F.3** rules out the literal reading of (i) for deterministic networks:
  noise-free $I(X;T)$ is constant while the layer stays injective on a
  discrete input set (it can fall only through exact collisions), and
  infinite when the representation is atomless. So "compression in mutual
  information" must mean compression of some estimate, or of a noisy
  surrogate.
- **1.F.4** supports (ii)'s mechanism. Saturation alone produces rise-then-fall
  in the tanh estimate, and the ReLU estimate grows or stays constant
  depending on the bins. But it also supports (iii): the estimate is a
  spread-at-resolution-$\Delta$ statistic, which is geometric, so "artefact"
  and "measures geometry" are both true.
- **1.F.5–1.F.6** show that once noise defines a scale, the drop in $I(X;T)$ at
  fixed label information is exactly within-class shrinkage. That is (iii)'s
  claim, made exact in a toy model (1.F.6.1). Whether mutual information is the right
  name for it is now a question about invariances (1.F.6.4). Noise-free
  mutual information is unchanged by any invertible map of $T$; a geometric
  statistic, such as the ratio of within-class to between-class scatter, is
  not. Two representations with equal estimated $I(X;T)$ and $I(T;Y)$ but
  different geometry would separate the two.
- None of 1.F.3–1.F.6 bears on causation (does clustering *cause*
  generalization?). That needs interventions: for example, penalize any
  decrease in within-class spread and see whether test loss suffers. Two
  observational checks are already in Saxe et al.: ReLU networks generalize
  without compressing in the binned estimate, and the compression phase also
  appears under full-batch training, which rules out SGD noise as its cause.
"""
    ),
]

IB_RD = [
    markdown(
        r"""
#### 1.F.2. The IB is rate–distortion with a learned distortion

**Why.** Rate–distortion (1.F.1) compresses a source $X$ into a representation
$T$ subject to a distortion measure $d(x,t)$ chosen in advance. The
*information bottleneck* (IB) of
[Tishby, Pereira and Bialek (1999)](https://arxiv.org/abs/physics/0004057)
replaces the distortion by a task: keep whatever information about $X$ is
relevant to a second variable $Y$ (a label), and discard the rest. Given the
joint law $p(x,y)$, it seeks the encoder $p(t\mid x)$ solving

$$
\min_{p(t\mid x)}\ I(X;T)-\beta\,I(T;Y),\qquad \beta>0,
$$

where $T$ is generated from $X$ alone, so $Y\to X\to T$ is a Markov chain.
Small $\beta$ favours compression; large $\beta$ favours relevance. This
exercise shows the IB is a rate–distortion problem whose distortion is a KL
divergence between predictions of $Y$, and that its optimal encoder has the
same Boltzmann form as in 1.F.1.6.

1. Define $p(y\mid t)=\sum_xp(y\mid x)\,p(x\mid t)$ and $d(x,t)=\mathrm{KL}\big(p(y\mid x)\,\|\,p(y\mid t)\big)$. Show that $\mathbb E\,d(X,T)=I(X;Y)-I(T;Y)$, so the IB Lagrangian is the rate–distortion Lagrangian $I(X;T)+\beta\,\mathbb E\,d(X,T)$ of 1.F.1 up to a constant.
2. Why is this not an ordinary rate–distortion problem?
3. Show that at a stationary point,
$$p(t\mid x)=\dfrac{p(t)}{Z(x,\beta)}\exp\!\big(-\beta\,\mathrm{KL}(p(y\mid x)\,\|\,p(y\mid t))\big),$$
i.e. 1.F.1.6 with $d=\mathrm{KL}$. Because $d$ now depends on the encoder, this needs one extra check: that the encoder-dependence of $d$ contributes nothing to the derivative of $\mathbb E\,d$.

<details><summary>Hint</summary>

For 3, differentiate $\sum_{x',t'}p(x')p(t'\mid x')\,d(x',t')$ with respect to $p(t\mid x)$. Differentiating the weight $p(t'\mid x')$ gives $p(x)\,d(x,t)$, as in 1.F.1.6. In the remaining term only $\log p(y\mid t)$ depends on the encoder; collect it as $-\sum_yp(t,y)\,\partial\log p(y\mid t)$ and use $\sum_yp(y\mid t)=1$.
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
using $p(x,t)p(y\mid x)=p(x,y,t)$, so that $p(y\mid t)$ as defined is the true
conditional. Hence $I(X;T)-\beta I(T;Y)=I(X;T)+\beta\,\mathbb E\,d-\beta I(X;Y)$,
and the last term does not depend on the encoder.

**2.** The distortion depends on the encoder itself, through $p(y\mid t)$. In
rate–distortion theory $d$ is fixed in advance. So the IB equations must be
solved self-consistently: alternate the Boltzmann update for $p(t\mid x)$ with
recomputing $p(t)$ and $p(y\mid t)$, as in Blahut–Arimoto (1.F.1.8). Unlike
there, the IB objective is not convex in the encoder, so the iteration finds a
local optimum.

**3.** Write $\mathbb E\,d=\sum_{x',t'}p(x')p(t'\mid x')\,d(x',t')$. Differentiating
the explicit weight $p(t\mid x)$ gives $p(x)\,d(x,t)$. The rest is
$$
\sum_{x'}p(x')p(t\mid x')\sum_yp(y\mid x')\Big(-\frac{\partial\log p(y\mid t)}{\partial p(t\mid x)}\Big)
=-\sum_yp(t,y)\,\frac{\partial\log p(y\mid t)}{\partial p(t\mid x)}
=-p(t)\,\frac{\partial}{\partial p(t\mid x)}\sum_yp(y\mid t)=0,
$$
because $\sum_yp(y\mid t)=1$ for every encoder (only the $t'=t$ terms depend on
$p(t\mid x)$ through $p(y\mid t')$). So $\partial\,\mathbb E\,d/\partial p(t\mid x)=p(x)\,d(x,t)$,
exactly as if $d$ were fixed, and the stationarity condition of 1.F.1.6,
$p(x)\big[\log\frac{p(t\mid x)}{p(t)}+\beta d(x,t)\big]+\lambda(x)=0$, gives the
stated form with $Z(x,\beta)$ the normalizer.

At large $\beta$, inputs $x$ are sent to the representations $t$ whose
prediction $p(y\mid t)$ is closest to their own $p(y\mid x)$: the IB clusters
inputs by what they say about $Y$.
"""
    ),
]
