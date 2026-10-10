"""WB2 §2: doesn't some theorem make this impossible?
"""
from nbkit import *  # noqa: F401,F403
from wb2_new_pacbayes import CELLS_VC as S2_VC, CELLS_PACBAYES as S2_PACBAYES  # noqa: F401  (2.C.1, 2.E)

S2_AB = [
    # ------------------------------------------------------------------ §2
    markdown(
        r"""
<a id="wb2-section-2"></a>

## 2. Doesn't some theorem make this impossible? (天无绝人之路)

Three classical theorems seem to say that learning from data should be
impossible, or at least unexplained. *No free lunch* says every learner is at
chance on the inputs it has not seen. *Universal approximation* says networks
can represent anything, so why do they learn this function rather than another?
*Capacity bounds* say generalization is guaranteed when the model class is
small, and the classes deep networks reach are enormous. Each theorem is true,
and none of them bites. This section proves each one in a simple setting and
finds the assumption that does the work. Then it reads the experiment that made
class-wide bounds untenable for deep learning (Zhang et al.), and introduces
PAC-Bayes, which bounds the error of the particular function training found
rather than the worst function in the class. 天无绝人之路, "heaven never closes
every road": the way out is a bet on structure in the targets, made implicitly
by the architecture and the training procedure.

**Notation.** A *hypothesis* $h$ is a predictor, and a *hypothesis class*
$\mathcal H$ is the set a learner chooses from. Examples $z=(x,y)$ are drawn
i.i.d. from an unknown distribution, and $S=(z_1,\dots,z_n)$ is the training
sample. For a loss $\ell(h;z)$, the *population risk* is
$R(h)=\mathbb E_z\,\ell(h;z)$, the expected loss on a fresh example, and the
*training risk* is $\hat R_n(h)=\frac1n\sum_{i=1}^n\ell(h;z_i)$. For
classification the loss is the 0–1 loss $\mathbf 1[h(x)\ne y]$, so $R(h)$ is
the error probability.

**What a learning guarantee looks like.** Most results here have the *PAC*
("probably approximately correct") form: *with probability at least
$1-\delta$* over the sample $S$, the learned predictor $\hat h$ has
$R(\hat h)\le\epsilon(n,\delta)$, or has a *generalization gap*
$R(\hat h)-\hat R_n(\hat h)\le\epsilon(n,\delta)$. The *sample complexity* is
the $n$ needed to make $\epsilon$ small. A bound is *vacuous* if $\epsilon$
exceeds the largest possible loss, so that it says nothing.

### 2.A. No free lunch

**The question.** Can any learning rule be better than another "in general"?
No free lunch says no: averaged over all possible targets, every learner does
equally badly on unseen inputs. The exercise proves it, finds the assumption
that does the work (a uniform prior over targets), and shows that a bet on
simple targets (an Occam bound) escapes it.

**Setup.** Let the input domain $\mathcal X$ be finite, with $|\mathcal X|=N$, and the
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

#### 2.A.1. Prove NFL, then find the bet

1. Let $f$ be uniform over all $2^N$ functions. Prove that every learner has expected off-training-set error exactly $\tfrac12$, conditional on any training inputs.
2. Conclude that for any two learners $A$ and $B$, $\sum_f\big[\mathrm{err}_{\mathrm{OTS}}(A,f)-\mathrm{err}_{\mathrm{OTS}}(B,f)\big]=0$. What does this say about a learner that beats another on some targets?
3. Now let $f$ lie in a class $\mathcal F$ with $|\mathcal F|=M$, and let the inputs be i.i.d. from a distribution $D$ on $\mathcal X$. Show that a learner returning any $h\in\mathcal F$ *consistent* with the sample (agreeing with $f$ on every training input) satisfies $\Pr[\mathrm{err}_D(h)>\epsilon]\le Me^{-n\epsilon}$, where $\mathrm{err}_D$ is the error under $D$. How many samples suffice if $\mathcal F$ is the set of functions computed by programs of at most $b$ bits? What does the bound give for the class of all functions?
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
- a symmetry prior, which favours equivariant architectures such as CNNs (WB3).

The question "why does deep learning generalize?" becomes: which prior over targets do the architecture and the optimizer implement, and why does it match the world?
"""
    ),
    markdown(
        r"""
### 2.B. Universal approximation: existence, not mechanism

**The question.** Universal approximation theorems (UAT) say that a large
enough network can approximate any reasonable function. Does that explain
anything about deep learning? The exercise counts how many units are needed in
one dimension, shows that depth can buy exponentially many linear pieces, and
lists what the theorem does not say.

**UAT (quoted).** For any continuous $f$ on a compact $K\subset\mathbb R^d$ and
any $\epsilon>0$, a one-hidden-layer network with a continuous non-polynomial
activation (for example ReLU) achieves $\sup_K|f-g|<\epsilon$. Cybenko (1989)
and Hornik (1991) proved the sigmoidal versions; the non-polynomial
characterization is
[Leshno, Lin, Pinkus and Schocken (1993)](https://linkinghub.elsevier.com/retrieve/pii/S0893608005801315).
The theorem says nothing about how many units are needed. The problem below
counts them in one dimension. Part 4 follows the depth-separation construction
of [Telgarsky (2015)](https://arxiv.org/abs/1509.08101).

#### 2.B.1. How many units for $\sin\omega x$?

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

**The question.** The classical guarantee of generalization: if the hypothesis
class is small compared with $n$, then training risk is close to population
risk *uniformly over the class*, so whatever the learner picks generalizes.
How small is "small" for an infinite class (VC dimension, 2.C.1), and what do
these bounds say about a real network (2.C.2)?

**Uniform convergence.** Two quantities measure how well a learned $\hat h$
did: the *generalization gap* $R(\hat h)-\hat R_n(\hat h)$ (test minus
training risk) and the *excess risk* $R(\hat h)-\inf_{h\in\mathcal H}R(h)$
(how far $\hat h$ is from the best member of the class). For empirical risk
minimization (ERM, $\hat h\in\arg\min_{\mathcal H}\hat R_n$), the excess risk
is at most $2\sup_{h\in\mathcal H}|R(h)-\hat R_n(h)|$ (proved in 1.A.1.3). So
it suffices to show that the gap is small *for every $h$ in the class at
once*, whatever the learner picks. For a finite class with losses in $[0,1]$,
Hoeffding's inequality and a union bound give, with probability at least
$1-\delta$,

$$
\sup_{h\in\mathcal H}|R(h)-\hat R_n(h)|\le\sqrt{\frac{\ln(2|\mathcal H|/\delta)}{2n}} ,
$$

which 2.C.2.1 derives. The size of the class enters only through
$\ln|\mathcal H|$, so the bound is useful when $n\gg\ln|\mathcal H|$. Classes
with real-valued parameters are infinite; for them 2.C.1 replaces
$\ln|\mathcal H|$ by a combinatorial size, the VC dimension. 2.C.2 then plugs
a real network into both bounds. Finite classes and VC dimension are in
chapters 4 and 6 of UML (Shalev-Shwartz and Ben-David, linked in 2.A).

"""
    ),
]

S2_C2 = [
    markdown(
        r"""
#### 2.C.2. Plug in a real network

How big are these bounds for a network of realistic size? Use the 0–1 loss
(so $R$ is test error and $\hat R_n$ training error) and the finite-class
bound $\sup_h|R(h)-\hat R_n(h)|\le\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$ from
the 2.C introduction.

1. Derive the finite-class bound from Hoeffding's inequality for one fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$, and a union bound.
2. A network with $W=10^6$ float32 weights is a finite class: each weight takes at most $2^{32}$ values. Evaluate the bound for $n=5\times10^4$ (the size of the CIFAR-10 training set) and $\delta=0.05$. Then compare with $n$ the VC-dimension scale $d\approx W\ln W$ (the general upper bound for ReLU networks of fixed depth, quoted in 2.C.1).
3. Zhang et al. (2.D) show that this kind of architecture, trained by SGD, fits CIFAR-10 images with uniformly random labels to zero training error. Let $\mathcal H$ be the set of functions the architecture and SGD can produce, and write $\mathrm{gap}(h)=R(h)-\hat R_n(h)$. Assume such interpolation succeeds with high probability on i.i.d. samples from a distribution with independent uniform labels and distinct inputs. Show that every bound of the form "for every data distribution $P$, with probability $\ge1-\delta$, all $h\in\mathcal H$ have $\mathrm{gap}(h)\le\epsilon(n,\delta)$" must have $\epsilon\ge0.9$ (10 classes), up to sampling fluctuation.

<details><summary>Hint</summary>

For 3, the bound has to hold for *every* $P$. Choose one.
</details>
"""
    ),
    answer(
        r"""
**1.** For a fixed $h$, $\Pr[|\hat R_n(h)-R(h)|>t]\le2e^{-2nt^2}$. A union bound over $|\mathcal H|$ hypotheses gives $\Pr[\sup_h|\cdot|>t]\le2|\mathcal H|e^{-2nt^2}$. Setting this equal to $\delta$ and solving gives $t=\sqrt{\ln(2|\mathcal H|/\delta)/(2n)}$.

**2.** Using the parameter-encoding upper bound $\ln|\mathcal H|\le32W\ln2\approx2.22\times10^7$, so the bound is $\sqrt{(2.22\times10^7+\ln 40)/10^5}\approx\sqrt{222}\approx14.9$. Since the 0–1 loss is at most 1, this parameter-encoding bound says nothing. The rough VC scale is $W\ln W\approx1.4\times10^7\gg n=5\times10^4$, and the VC bound needs $n\gg d$, so it is vacuous too. This scale is a generic upper bound; it does not prove that this architecture has that VC dimension. But if the actual VC dimension is at least $n$, NFL bites as well: the class shatters some $n$ points, and for $D$ uniform on those points the sample misses an expected fraction $(1-1/n)^n\approx e^{-1}$ of them, so by 2.A.1.1 some target forces expected error at least $\tfrac12(1-1/n)^n\approx0.18$ on any learner.

**3.** Take $P'$ to be CIFAR-10 images with labels drawn uniformly from the 10 classes, independently of the image. Assume the random-label fitting result extends to i.i.d. samples from this input distribution, with distinct inputs and high probability of interpolation. Then SGD reaches an $h\in\mathcal H$ with $\hat R_n(h)=0$. Its labels are independent of the input, so $R(h)=0.9$. So on $P'$, some $h\in\mathcal H$ has gap $0.9$ with high probability. A bound that depends only on $\mathcal H$, $n$ and $\delta$ must hold for $P'$, so $\epsilon(n,\delta)\ge0.9-o(1)$. The same distribution-independent bound is then vacuous for the real labels as well. The experiment on one fixed image collection motivates the interpolation assumption; it does not prove it for every random draw. Vacuous is not wrong, but it cannot be the explanation. 2.D.1.1 asks which explanations survive.
"""
    ),
]

S2_D = [
    markdown(
        r"""
### 2.D. Reading: networks fit random labels

**Why.** This is the experiment that made class-wide capacity bounds untenable
as an explanation of deep learning. Standard networks, trained the standard
way, fit labels that carry no information at all. Whatever class they range
over is therefore large enough to make every bound of 2.C vacuous, yet with
real labels the same networks generalize.

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
            importance=5, length=2,
        )
        + r"""

#### 2.D.1. Questions on Zhang et al.

1. List the explanations of generalization that the random-labels experiment rules out, and those it leaves standing.
2. The paper finds that explicit regularizers (weight decay, dropout, data augmentation) help but are neither necessary nor sufficient. Which of the three views of ridge regularization in 1.D (prior, constraint, spectral filter) does this count against, and which survive? (This uses 1.D.1, 1.D.2 and 1.D.4.)
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
- norm-, margin-, compression- and PAC-Bayes-style bounds evaluated on the learned function (2.E);
- a prior over functions whose support is everything but whose mass is on structured functions.

The slower fit on random labels is a hint that some simplicity ordering is at work. Such a "soft" bias (the class can fit anything, but training prefers some fits) predicts things a hard capacity restriction does not:
- the same model at the same $n$ generalizes differently depending on the labels;
- random labels take longer to fit and end at larger norm or complexity;
- adding parameters need not hurt, and can help (double descent, section 3), whereas a class-wide capacity upper bound can worsen with class size without predicting that actual risk must worsen;
- a bound in terms of the prior mass or description length of the *found* function can be non-vacuous where class-capacity bounds are not.

**2.** It counts against the *constraint* view used as an explanation, in which a norm ball is small enough for its capacity to explain generalization. The nets generalize without the constraint, and with the constraint they still fit random labels, so the constrained class is still huge. The *prior* view survives if the prior is soft and implemented implicitly, by architecture, initialization and SGD. The *spectral-filter* view also survives as implicit regularization: early stopping and gradient descent's preference for large-eigenvalue directions, and gradient descent's convergence to the minimum-norm solution (1.D.4).
"""
    ),
]

