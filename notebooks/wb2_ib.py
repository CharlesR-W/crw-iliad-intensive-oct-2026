"""WB2 §9: did deep networks compress? The information-bottleneck debate.

Formulas are checked by checks/wb2b_check.py (old 1.F.3-1.F.4).
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
<a id="wb2-section-9"></a>

## 9. Did deep networks compress? The information bottleneck

A once-popular account said networks generalize because SGD compresses what
their layers keep about the input; the debate that followed is a compact
lesson in what mutual information can and cannot measure about a
deterministic network.

Notation: $X$ is the input, $T=f(X)$ a layer's activations, $I(X;T)$ their mutual information, $q_\Delta$ a discretization of $T$ with bin width $\Delta$ and $K$ bins.
"""
        + reading(
            "Opening the black box of deep neural networks via information",
            "https://arxiv.org/abs/1703.00810",
            "Ravid Shwartz-Ziv, Naftali Tishby (arXiv, 2017)",
            "The claim: in the information plane, training has a short fitting phase and a long compression "
            "phase driven by SGD noise. Read it against 9.1–9.2.",
            importance=4, length=2,
        )
        + reading(
            "On the information bottleneck theory of deep learning",
            "https://openreview.net/forum?id=ry_WPG-A-",
            "Andrew M. Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, "
            "Brendan D. Tracey, David D. Cox (ICLR 2018)",
            "The critique: compression appears with saturating tanh units but not ReLU, non-compressing "
            "networks generalize, and full-batch training shows the same curves.",
            importance=4, length=3,
        )
        + reading(
            "Estimating information flow in deep neural networks",
            "https://arxiv.org/abs/1810.05728",
            "Ziv Goldfeld, Ewout van den Berg, Kristjan Greenewald, Igor Melnyk, Nam Nguyen, "
            "Brian Kingsbury, Yury Polyanskiy (ICML 2019)",
            "Add a little noise to each layer and I(X;T) becomes finite; its decrease is then the clustering "
            "of same-class representations, which binned estimates also track.",
            importance=3, length=2,
        )
        + reading(
            "Deep variational information bottleneck",
            "https://arxiv.org/abs/1612.00410",
            "Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy (ICLR 2017)",
            "The IB objective as a trainable loss for a network with a stochastic encoder, with better "
            "robustness than deterministic baselines.",
            importance=2, length=2,
        )
        + r"""
Also: [Tishby, Pereira and Bialek (1999)](https://arxiv.org/abs/physics/0004057), the information bottleneck as rate–distortion with a learned distortion.
"""
        + references(
            book("Elements of Information Theory, 2nd ed.", "https://doi.org/10.1002/047174882X",
                 "Cover and Thomas (2006)", "ch. 2; ch. 10, §10.1–10.3"),
        )
        + r"""
### Exercises: what the information plane measures

**Setup for 9.1–9.2.** $X$ is the network input, $Y$ the label, and $T=f(X)$
the activations of one hidden layer, a deterministic function of $X$. The
*information plane* plots each layer as $\big(I(X;T),\,I(T;Y)\big)$ during
training; moving left is "compression". For discrete variables,
$H(X)=-\sum p\log p$, $I(X;T)=H(T)-H(T\mid X)$, and the *data-processing
inequality* says that if $X\to T\to T'$ is a Markov chain (e.g. $T'$ a
function of $T$), then $I(X;T')\le I(X;T)$. Quantizing a variable with density
$g$ to bins of small width $\delta$ gives entropy $h-\log\delta+o(1)$, where
$h=-\int g\log g$ is its differential entropy. Shwartz-Ziv and Tishby estimated
$I(X;T)$ by binning a $\tanh$ network's activations into 30 equal bins on
$[-1,1]$, with $X$ uniform over 4096 inputs.

**9.1. A deterministic network has nothing to compress.**

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
**9.2. What the binned estimator measures: saturation looks like
compression.** The binned estimator replaces $T$ by its bin index
$\mathrm{bin}(T)$ and reports $I(X;\mathrm{bin}(T))$, which by 9.1.1 is just
$H(\mathrm{bin}(T))$. Take one unit with standard normal pre-activation $Z$
and pre-activation scale $a>0$, the weight norm. Training typically grows $a$.

1. $T=\tanh(aZ)$, binned into $K$ equal bins on $[-1,1]$, with $K$ even. Show that $H(\mathrm{bin}(T))\to\log2$ as $a\to\infty$. What is the limit as $a\to0$? How does it change if $K$ is odd?
2. Sketch $H(\mathrm{bin}(T))$ as a function of $a$ for $K=30$. Explain why steadily growing weights alone produce a "fitting phase" followed by a "compression phase".
3. $T=\mathrm{ReLU}(aZ)=\max(0,aZ)$, binned with a fixed width $\Delta$: $[0,\Delta),[\Delta,2\Delta),\dots$. Show that as $a\to\infty$,
$$
H(\mathrm{bin}(T))=\log2+\tfrac12\Big(h(|Z|)+\log\frac a\Delta\Big)+o(1),\qquad h(|Z|)=\tfrac12\log\frac{\pi e}2 .
$$
So the estimate grows without bound, like $\tfrac12\log a$.
4. Many implementations instead bin ReLU activations into $K$ equal bins on $[0,\max T]$, the observed range. What happens to the estimate as $a$ grows?
5. In one sentence: what quantity is the binned estimator measuring?
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
]
