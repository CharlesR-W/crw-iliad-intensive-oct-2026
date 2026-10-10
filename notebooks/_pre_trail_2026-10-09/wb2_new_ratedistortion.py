"""WB2 new material: information bottleneck and rate–distortion (opening of 1.F).

Pitch, information-theory primer, and 1.F.1 (rate–distortion warm-ups).
Old 6.E follows as 1.F.2; old 6.A–6.D and 6.F follow from 1.F.3 on.
Formulas are checked numerically in checks/wb2_new_ratedistortion_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
### 1.F. Information bottleneck and rate–distortion

**The question.** A common intuition about learned representations is that a
good one is a *compressed* summary of the input: it throws away what is
irrelevant to the task and keeps what is relevant. On this view the last
hidden layer of an image classifier should have forgotten the lighting and the
background and remembered the object. The intuition is appealing, but it says
nothing definite until "compressed" and "relevant" are given a meaning. This
subsection gives them one, using information theory, and then looks at a
debate over whether deep networks actually compress during training.

**Rate–distortion theory** is Shannon's theory of lossy compression (1959).
Suppose a source $X$ may be described only approximately, with average error
at most $D$ under a chosen distortion measure $d$. What is the fewest nats per
symbol that such a description can use? The answer, $R(D)$, is the solution
of a mutual-information minimization. Its optimal encoders have a Boltzmann
form, with the distortion as energy and an inverse temperature $\beta$.
1.F.1 works through the two textbook sources (a biased coin and a Gaussian)
and the general solution.

**The information bottleneck** (IB, 1.F.2) applies the same idea with the
distortion set by a task: a representation $T$ of $X$ should keep the
information about a target $Y$ and discard the rest. It turns out to be a
rate–distortion problem whose distortion measure depends on the encoder being
optimized.

**The information plane.** Do trained networks actually compress in this
sense? Shwartz-Ziv and Tishby (2017) reported that hidden layers first fit the
labels and then compress, and connected the compression to generalization.
The exchange that followed turned on what mutual information measures when a
layer is a deterministic function of its input. 1.F.3–1.F.6 derive the points
at issue in small models; 1.F.7 has the papers.

**What is here.** A primer (below) defines every information-theoretic quantity
used in the subsection; skip it if you know Cover and Thomas ch. 2. 1.F.1 and
1.F.2 are classical information theory. The information-plane material
(1.F.3–1.F.7) needs only the primer, so you can go straight there if the
deep-learning debate is what interests you.

**Notation for the whole subsection.** $X$ is the input, with distribution
$p(x)$; $Y$ is the target or label; $T$ is a representation of $X$, produced
by an *encoder* $p(t\mid x)$, a conditional distribution. A deterministic
encoder $T=f(X)$, such as a hidden layer of a network, is the special case
where $p(t\mid x)$ puts all its mass on $f(x)$. Because $T$ is computed from $X$
alone (plus, possibly, independent noise), the three variables form a Markov
chain $Y\to X\to T$, defined below.
"""
    ),
    markdown(
        r"""
**Information-theory primer.** All logarithms are natural, so information is
measured in nats; $1$ nat $=1/\log 2\approx1.44$ bits.

*Entropy.* For a discrete random variable $X$ with probability mass function
$p(x)$,
$$
H(X)=-\sum_xp(x)\log p(x)=\mathbb E\big[-\log p(X)\big].
$$
It satisfies $0\le H(X)\le\log|\mathcal X|$, where $\mathcal X$ is the set of
values. It is $0$ iff $X$ is deterministic and maximal iff $X$ is uniform.
Operationally, $H(X)$ is the minimum average number of nats per symbol needed
to encode a long i.i.d. sequence of copies of $X$ without loss (Shannon's
source-coding theorem). The entropy of a Bernoulli($q$) variable is the
*binary entropy*
$$
H_b(q)=-q\log q-(1-q)\log(1-q),
$$
which increases on $[0,\tfrac12]$ from $0$ to $\log2$ and is symmetric about
$\tfrac12$.

*Joint and conditional entropy.* $H(X,Y)$ is the entropy of the pair. The
conditional entropy is
$H(Y\mid X)=-\sum_{x,y}p(x,y)\log p(y\mid x)=\sum_xp(x)\,H(Y\mid X=x)$, the
average uncertainty left in $Y$ once $X$ is known. Chain rule:
$H(X,Y)=H(X)+H(Y\mid X)$. Conditioning does not increase entropy on average:
$H(Y\mid X)\le H(Y)$. If $T=f(X)$ then $H(T\mid X)=0$.

*Relative entropy.* For distributions $p,q$ on the same set,
$$
\mathrm{KL}(p\,\|\,q)=\sum_xp(x)\log\frac{p(x)}{q(x)}\ \ge0,
$$
with equality iff $p=q$ (Gibbs' inequality, which is Jensen's inequality
applied to $-\log$). It is not symmetric, and it is $+\infty$ if $p$ puts mass
where $q$ does not.

*Mutual information.* The information that $X$ and $Y$ carry about each other
has several equivalent forms:
$$
I(X;Y)=\mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)
=H(X)+H(Y)-H(X,Y).
$$
It is symmetric, nonnegative, zero iff $X$ and $Y$ are independent, and
$I(X;X)=H(X)$. A fifth form is often the most useful one:
$I(X;Y)=\sum_xp(x)\,\mathrm{KL}\big(p(y\mid x)\,\|\,p(y)\big)$, the average
amount by which learning $x$ moves the distribution of $Y$. Conditional mutual
information is $I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z)\ge0$, and the chain rule
for information is $I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)$.

*Differential entropy.* For a random vector $X$ with density $f$ on
$\mathbb R^d$, $h(X)=-\int f\log f$. It is *not* the limit of discrete
entropies: quantizing $X$ to cubes of side $\Delta$ gives an entropy of about
$h(X)-d\log\Delta$, which diverges as $\Delta\to0$. Unlike $H$, $h$ can be
negative and depends on units: $h(aX)=h(X)+d\log|a|$, and more generally
$h(AX)=h(X)+\log|\det A|$. It is unchanged by translation, $h(X+c)=h(X)$. For
a Gaussian,
$$
h\big(\mathcal N(\mu,\sigma^2)\big)=\tfrac12\log(2\pi e\sigma^2),\qquad
h\big(\mathcal N(\mu,\Sigma)\big)=\tfrac12\log\det(2\pi e\,\Sigma).
$$
*Maximum entropy:* among all densities with covariance $\Sigma$, the Gaussian
has the largest $h$. In one dimension, any $W$ with $\mathbb EW^2\le s$ has
$h(W)\le\tfrac12\log(2\pi e s)$. Conditional differential entropy
$h(X\mid Y)$ is defined as in the discrete case.

*Mutual information for continuous variables.* When densities exist,
$I(X;Y)=h(X)-h(X\mid Y)=\mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)$. Unlike
$h$, mutual information does not depend on units: it is unchanged by any
invertible transformation of $X$, or of $Y$, separately. It equals the
supremum of the discrete mutual information between quantizations of $X$ and
of $Y$. It can be $+\infty$; for example $I(X;X)=\infty$ when $X$ has a
density.

*Markov chains and data processing.* $Y\to X\to T$ means
$p(y,x,t)=p(y)\,p(x\mid y)\,p(t\mid x)$: given $X$, the variable $T$ carries no
further information about $Y$, i.e. $I(Y;T\mid X)=0$. This holds whenever $T$
is computed from $X$ alone, possibly with independent noise. The chain is
reversible: $T\to X\to Y$ holds too. The *data-processing inequality* says
$$
I(Y;T)\le I(Y;X):
$$
processing $X$ cannot create information about $Y$. Proof: expand $I(Y;X,T)$
by the chain rule in two orders,
$I(Y;X)+I(Y;T\mid X)=I(Y;T)+I(Y;X\mid T)$, and use $I(Y;T\mid X)=0$ and
$I(Y;X\mid T)\ge0$. Applied to the reversed chain, it also gives
$I(T;Y)\le I(T;X)$. So the information-plane point $\big(I(X;T),I(T;Y)\big)$
of any representation satisfies $I(T;Y)\le\min\big(I(X;T),\,I(X;Y)\big)$.
"""
        + references(
            book(
                "Elements of Information Theory, 2nd ed.",
                "https://doi.org/10.1002/047174882X",
                "Cover and Thomas (2006)",
                "ch. 2; ch. 10, §10.1–10.3 and §10.8",
            ),
        )
    ),
    markdown(
        r"""
#### 1.F.1. Rate–distortion warm-ups

**Why.** Before compressing for a task (1.F.2), compress for fidelity. The
two textbook sources show exactly what an optimal lossy description keeps and
what it throws away, and the general solution (parts 6–8) has the Boltzmann
form that the information bottleneck inherits.

**Setup.** A source emits $X\sim p(x)$. An encoder $p(t\mid x)$ represents it by
$T$, which takes values in a *reproduction alphabet* $\mathcal T$ (often the
same set as the values of $X$). A *distortion measure* $d(x,t)\ge0$ says how
bad it is to represent $x$ by $t$: Hamming distortion
$d(x,t)=\mathbf 1[x\ne t]$ for bits, squared error $d(x,t)=(x-t)^2$ for real
numbers. The *rate–distortion function* is
$$
R(D)=\min_{p(t\mid x):\ \mathbb E\,d(X,T)\le D}\ I(X;T),
$$
with the expectation over $p(x)\,p(t\mid x)$. $I(X;T)$ is how many nats about
$X$ the representation keeps, so $R(D)$ is the least that any representation
accurate to $D$ on average must keep.

**What $R(D)$ means** (Shannon 1959; Cover and Thomas ch. 10; quoted, not
proved here). Code long i.i.d. blocks $X_1,\dots,X_n$ jointly, and ask for an
average per-symbol distortion of at most $D$. Then any rate above $R(D)$ nats
per symbol is achievable as $n\to\infty$, and no rate below $R(D)$ is. So
$R(D)$ is the minimum number of nats per symbol needed for lossy coding at
distortion $D$. A single symbol cannot be coded at a fractional rate; the
theorem is about blocks. For Hamming distortion, $R(0)=H(X)$, the lossless
limit.

Two facts you may use freely. $R(D)$ is non-increasing, because a larger $D$
allows more encoders. It is convex, because $I(X;T)$ is a convex function of
$p(t\mid x)$ at fixed $p(x)$ while $\mathbb E\,d$ is linear in it, so mixing
two encoders gives at most the mixed rate at the mixed distortion. Also,
$R(D)=0$ once $D\ge D_{\max}=\min_t\mathbb E\,d(X,t)$, since a constant
representation achieves $D_{\max}$.

1. *Bernoulli source, lower bound.* Let $X\sim\mathrm{Bernoulli}(p)$ with $p\le\tfrac12$, $\mathcal T=\{0,1\}$, and Hamming distortion. Show that every encoder with $\mathbb E\,d(X,T)\le D$, where $D\le p$, has $I(X;T)\ge H_b(p)-H_b(D)$.
2. *Bernoulli source, achievability.* Build the joint law backwards: $T\sim\mathrm{Bernoulli}(r)$, $Z\sim\mathrm{Bernoulli}(D)$ independent of $T$, and $X=T\oplus Z$ (addition mod 2). Find the $r$ that makes $X\sim\mathrm{Bernoulli}(p)$ and check that it lies in $[0,1]$ for $0\le D\le p$. Show that this joint law has $\mathbb E\,d=D$ and $I(X;T)=H_b(p)-H_b(D)$. Conclude that $R(D)=H_b(p)-H_b(D)$ for $0\le D\le p$ and $R(D)=0$ for $D\ge p$.
3. Take $p=\tfrac12$ and work in bits. What are $R(0)$ and $R'(D)$ as $D\to0$, and what does the slope mean? How much rate does tolerating 1% bit errors save, compared with lossless coding?
4. *Gaussian source, lower bound.* Let $X\sim\mathcal N(0,\sigma^2)$, $\mathcal T=\mathbb R$, and squared-error distortion. Show that every encoder with $\mathbb E(X-T)^2\le D\le\sigma^2$ has $I(X;T)\ge\tfrac12\log(\sigma^2/D)$.
5. *Gaussian source, achievability.* Let $T\sim\mathcal N(0,\sigma^2-D)$ and $Z\sim\mathcal N(0,D)$ be independent, and $X=T+Z$. Show that this achieves the bound, so $R(D)=\tfrac12\log(\sigma^2/D)$ for $D\le\sigma^2$ and $0$ beyond. Find the forward encoder $p(t\mid x)$. Invert to $D(R)$ in bits: how much does each extra bit reduce the mean squared error?

<details><summary>Hint</summary>

For 1, write $I(X;T)=H(X)-H(X\mid T)$. Given $T$, the variables $X$ and $X\oplus T$ determine each other, so $H(X\mid T)=H(X\oplus T\mid T)$; then drop the conditioning. Note $\Pr[X\oplus T=1]=\mathbb E\,d$. For 4, the same argument with $X-T$ in place of $X\oplus T$, plus the maximum-entropy property of the Gaussian. For 5, $(X,T)$ is jointly Gaussian; condition $T$ on $X$ by the usual Gaussian formulas.
</details>
"""
    ),
    answer(
        r"""
**1.** $I(X;T)=H(X)-H(X\mid T)=H_b(p)-H(X\mid T)$. Given $T=t$, the map
$x\mapsto x\oplus t$ is a bijection, so $H(X\mid T)=H(X\oplus T\mid T)\le H(X\oplus T)$.
The variable $X\oplus T$ is Bernoulli with parameter
$\Pr[X\ne T]=\mathbb E\,d\le D\le p\le\tfrac12$, and $H_b$ is increasing on
$[0,\tfrac12]$, so $H(X\oplus T)=H_b(\mathbb E\,d)\le H_b(D)$. Hence
$I(X;T)\ge H_b(p)-H_b(D)$.

**2.** $\Pr[X=1]=r(1-D)+(1-r)D=D+r(1-2D)$, so $r=\dfrac{p-D}{1-2D}$ (for $D<\tfrac12$).
It is $\ge0$ because $D\le p$, and $\le1$ because $p-D\le1-2D$ is $p+D\le1$.
The distortion is $\Pr[X\ne T]=\Pr[Z=1]=D$. Since $Z$ is independent of $T$,
$H(X\mid T)=H(T\oplus Z\mid T)=H(Z)=H_b(D)$, so $I(X;T)=H_b(p)-H_b(D)$. The
encoder is the forward conditional $p(t\mid x)=p(t)p(x\mid t)/p(x)$; only the
joint law matters for $I$ and $\mathbb E\,d$. Together with 1, this gives
$R(D)=H_b(p)-H_b(D)$ on $[0,p]$. For $D\ge p$, the constant $T=0$ has
$\mathbb E\,d=\Pr[X=1]=p\le D$ and $I=0$.

**3.** In bits, $R(D)=1-H_2(D)$ on $[0,\tfrac12]$, where $H_2$ is binary
entropy in bits. $R(0)=1$ bit: zero distortion forces $T$ to determine $X$, so
$I(X;T)=H(X)$, the lossless rate. The slope is
$R'(D)=-H_b'(D)=-\log\frac{1-D}{D}\to-\infty$ as $D\to0$ (in nats; divide by
$\log2$ for bits): the first errors you tolerate save the most rate. At
$D=0.01$, $R=1-H_2(0.01)\approx1-0.081=0.919$ bits, so 1% bit errors save
about 8% of the rate. At $D=0.1$ the rate is already down to $0.531$ bits.

**4.** $I(X;T)=h(X)-h(X\mid T)$. Translation does not change differential
entropy, so $h(X\mid T)=h(X-T\mid T)\le h(X-T)$. The variable $X-T$ has
$\mathbb E(X-T)^2\le D$, so by maximum entropy $h(X-T)\le\tfrac12\log(2\pi eD)$.
With $h(X)=\tfrac12\log(2\pi e\sigma^2)$, this gives
$I(X;T)\ge\tfrac12\log(\sigma^2/D)$.

**5.** $X=T+Z$ is Gaussian with variance $(\sigma^2-D)+D=\sigma^2$, as required.
$\mathbb E(X-T)^2=\mathbb EZ^2=D$, and
$I(X;T)=h(X)-h(X\mid T)=h(X)-h(Z)=\tfrac12\log(\sigma^2/D)$. With 4, this is
$R(D)$. For $D\ge\sigma^2$, $T=0$ gives distortion $\sigma^2$ at zero rate.

Forward encoder: $\mathrm{Cov}(X,T)=\mathrm{Var}\,T=\sigma^2-D$, so
$$
T\mid X=x\ \sim\ \mathcal N\Big(\big(1-\tfrac{D}{\sigma^2}\big)x,\ \ D\big(1-\tfrac{D}{\sigma^2}\big)\Big).
$$
The optimal encoder shrinks $x$ toward zero by the factor $1-D/\sigma^2$ and
adds independent Gaussian noise. In bits, $R=\tfrac12\log_2(\sigma^2/D)$, i.e.
$D(R)=\sigma^2\,2^{-2R}$: each extra bit divides the mean squared error by 4
(about 6 dB per bit).

Note that this optimal single-symbol encoder is noisy, and its $I(X;T)$ is
finite. A noiseless invertible encoder such as $T=X$ would have
$I(X;T)=\infty$. The block codes that achieve $R(D)$ in the operational sense
are deterministic quantizers with finitely many codewords, so their mutual
information is finite too. The question of what $I(X;T)$ means for a
deterministic map with continuous output returns from 1.F.3 on.
"""
    ),
    markdown(
        r"""
**The Lagrangian form.** Because $R(D)$ is convex, the constrained problem can
be traded for an unconstrained one: for $\beta\ge0$, minimize
$$
\mathcal F[p(t\mid x)]=I(X;T)+\beta\,\mathbb E\,d(X,T)
$$
over encoders, where $I(X;T)$ is computed with the marginal
$p(t)=\sum_xp(x)\,p(t\mid x)$. Each $\beta$ picks out one point
$(D_\beta,R_\beta)$ of the curve. Part 6 is the standard characterization of
the optimal encoder (Cover and Thomas §10.7); 1.F.2.3 reuses it for the
information bottleneck. Parts 7–8 interpret it and turn it into an algorithm.
Part 9 is optional.

6. *Boltzmann form.* Show that at a stationary point of $\mathcal F$,
$$
p(t\mid x)=\frac{p(t)\,e^{-\beta d(x,t)}}{Z(x,\beta)},\qquad Z(x,\beta)=\sum_tp(t)\,e^{-\beta d(x,t)} .
$$
Show also that at such a point $I(X;T)+\beta\,\mathbb E\,d=-\mathbb E_X\log Z(X,\beta)$. Interpret $d$, $\beta$, $p(t)$ and $\log Z$ in statistical-mechanics terms, and describe the encoder as $\beta\to0$ and $\beta\to\infty$. Why is this not yet a closed-form solution?
7. *$\beta$ is the slope.* Let $F(\beta)=R_\beta+\beta D_\beta$ be the minimum value of $\mathcal F$. Using the envelope theorem, $F'(\beta)=D_\beta$, show that along the curve $dR/dD=-\beta$. Check this against parts 2 and 5: write the test channels found there in Boltzmann form, read off $\beta$, and compare with $-R'(D)$. What is the optimal encoder when $\beta$ is below $\log\frac{1-p}{p}$ (Bernoulli) or $\frac1{2\sigma^2}$ (Gaussian)?
8. *Alternating minimization (Blahut–Arimoto).* For any distribution $q(t)$, show that
$$
\sum_xp(x)\,\mathrm{KL}\big(p(\cdot\mid x)\,\|\,q\big)=I(X;T)+\mathrm{KL}(p_T\,\|\,q),
$$
where $p_T$ is the marginal of $T$. Conclude that minimizing $\mathcal F$ is the same as minimizing $G(p(t\mid x),q)=\sum_xp(x)\,\mathrm{KL}\big(p(\cdot\mid x)\,\|\,q\big)+\beta\,\mathbb E\,d$ over encoder and $q$ jointly. Show that minimizing $G$ over the encoder at fixed $q$ gives $p(t\mid x)\propto q(t)e^{-\beta d(x,t)}$, and over $q$ at fixed encoder gives $q=p_T$. Why does alternating these two steps never increase $G$?
9. *(Optional) Reverse water-filling.* Let $X=(X_1,\dots,X_k)$ have independent components $X_i\sim\mathcal N(0,\sigma_i^2)$, with $d(x,t)=\|x-t\|^2$. Show that
$$
R(D)=\sum_i\tfrac12\log\frac{\sigma_i^2}{D_i},\qquad D_i=\min(\theta,\sigma_i^2),\quad\text{with }\theta\text{ fixed by }\textstyle\sum_iD_i=D .
$$
Which components are described, and which are dropped? Why does the same answer hold for a correlated Gaussian $X\sim\mathcal N(0,\Sigma)$, with $\sigma_i^2$ replaced by the eigenvalues of $\Sigma$? Compare the optimal encoder with principal component analysis, and with ridge regression, which multiplies the least-squares coefficient along the $k$-th principal direction of the inputs by $s_k^2/(s_k^2+n\lambda)$ (here $s_k^2$ is the $k$-th eigenvalue of $X^\top X$ for an $n\times p$ design matrix, and $\lambda$ the ridge penalty).

<details><summary>Hint</summary>

For 6, differentiate $I(X;T)=\sum_{x,t}p(x)p(t\mid x)\log\frac{p(t\mid x)}{p(t)}$ with respect to $p(t\mid x)$, remembering that $p(t)$ depends on it; the terms from differentiating $p(t)$ cancel, leaving $p(x)\log\frac{p(t\mid x)}{p(t)}$. Add a multiplier $\lambda(x)$ for $\sum_tp(t\mid x)=1$. For 7, in the Bernoulli case $p(x\mid t)=(1-D)\,e^{-\beta d(x,t)}$ with $e^{-\beta}=D/(1-D)$. For 9, first show $I(X;T)\ge\sum_iI(X_i;T_i)$ using independence of the $X_i$ and "conditioning reduces entropy"; then apply part 4 to each component and minimize over the allocation $D_i$.
</details>
"""
    ),
    answer(
        r"""
**6.** Write $I(X;T)=\sum_{x,t}p(x)p(t\mid x)\big[\log p(t\mid x)-\log p(t)\big]$
with $p(t)=\sum_{x'}p(x')p(t\mid x')$. Then
$$
\frac{\partial I}{\partial p(t\mid x)}=p(x)\log\frac{p(t\mid x)}{p(t)}+p(x)-\sum_{x'}p(x')p(t\mid x')\,\frac{p(x)}{p(t)}
=p(x)\log\frac{p(t\mid x)}{p(t)},
$$
and $\partial(\beta\,\mathbb E\,d)/\partial p(t\mid x)=\beta\,p(x)\,d(x,t)$.
With a multiplier $\lambda(x)$ for normalization, stationarity reads
$p(x)\big[\log\frac{p(t\mid x)}{p(t)}+\beta d(x,t)\big]+\lambda(x)=0$, so
$p(t\mid x)=p(t)e^{-\beta d(x,t)}\times(\text{a function of }x)$, and
normalizing gives $Z(x,\beta)$. Substituting
$\log\frac{p(t\mid x)}{p(t)}=-\beta d(x,t)-\log Z(x,\beta)$ into $I$ gives
$I(X;T)=-\beta\,\mathbb E\,d-\mathbb E\log Z(X,\beta)$, which is the stated
identity.

For each input $x$, the encoder is a Boltzmann distribution over the "states"
$t$, with energy $d(x,t)$, inverse temperature $\beta$, and a prior weight
$p(t)$ playing the role of a degeneracy. $-\log Z(x,\beta)$ is $\beta$ times
a free energy, and $\mathcal F$ at the optimum is its average over the
quenched input $x$. As $\beta\to0$ (high temperature), $p(t\mid x)\to p(t)$:
$T$ is independent of $X$ and the rate is zero. As $\beta\to\infty$ (zero
temperature), $p(t\mid x)$ concentrates on $\arg\min_td(x,t)$ among the $t$
with $p(t)>0$: a deterministic nearest-neighbour quantizer. It is not a closed
form because $p(t)$ on the right is the marginal of the encoder on the left;
the equation must be solved self-consistently (part 8).

**7.** At the minimizing encoder, changing $\beta$ changes $\mathcal F$ only
through its explicit $\beta$, because the encoder is stationary; so
$F'(\beta)=D_\beta$. Then $R_\beta=F(\beta)-\beta D_\beta$ gives
$dR_\beta/d\beta=D_\beta-D_\beta-\beta\,dD_\beta/d\beta$, i.e.
$dR/dD=-\beta$. So $\beta$ is the exchange rate between rate and distortion,
in nats per unit of distortion, and the convexity of $R(D)$ says that it
falls as $D$ grows.

*Bernoulli.* In part 2, $p(x\mid t)=1-D$ if $x=t$ and $D$ otherwise, i.e.
$p(x\mid t)=(1-D)e^{-\beta d(x,t)}$ with $\beta=\log\frac{1-D}{D}$. Then
$p(t\mid x)=p(t)p(x\mid t)/p(x)\propto p(t)e^{-\beta d(x,t)}$, the Boltzmann
form. And $-R'(D)=H_b'(D)=\log\frac{1-D}{D}$. They agree.

*Gaussian.* In part 5, $p(x\mid t)\propto e^{-(x-t)^2/(2D)}$, so
$p(t\mid x)\propto p(t)e^{-\beta(x-t)^2}$ with $\beta=\frac1{2D}$, and
$-R'(D)=\frac1{2D}$. They agree.

*Low $\beta$.* As $D$ rises to $D_{\max}$ (which is $p$ or $\sigma^2$), the
slope $\beta=-R'(D)$ falls to $\beta_c=\log\frac{1-p}{p}$ or
$\frac1{2\sigma^2}$. For $\beta\le\beta_c$, convexity gives
$R'(D)+\beta\le0$ for all $D<D_{\max}$, and $R=0$ beyond, so $R(D)+\beta D$
is minimized at $D=D_{\max}$: the optimal encoder is a constant, $T$ carries no
information, and $R_\beta=0$. Below a critical inverse temperature the
representation is trivial, a phase transition. For a fair coin, $\beta_c=0$.

**8.**
$\sum_xp(x)\sum_tp(t\mid x)\log\frac{p(t\mid x)}{q(t)}
=\sum_{x,t}p(x,t)\Big[\log\frac{p(t\mid x)}{p(t)}+\log\frac{p(t)}{q(t)}\Big]
=I(X;T)+\mathrm{KL}(p_T\,\|\,q)$.
This is at least $I(X;T)$, with equality iff $q=p_T$. So
$\min_{\text{enc}}\mathcal F=\min_{\text{enc}}\min_qG$.

At fixed $q$, $G$ splits over $x$, and for each $x$ we minimize
$\sum_tp(t\mid x)\big[\log\frac{p(t\mid x)}{q(t)}+\beta d(x,t)\big]
=\mathrm{KL}\big(p(\cdot\mid x)\,\big\|\,q\,e^{-\beta d(x,\cdot)}/Z_q(x)\big)-\log Z_q(x)$,
with $Z_q(x)=\sum_tq(t)e^{-\beta d(x,t)}$. This is smallest when the KL term
vanishes, at $p(t\mid x)=q(t)e^{-\beta d(x,t)}/Z_q(x)$ (the Gibbs variational
principle). At fixed encoder, only $\mathrm{KL}(p_T\|q)$ depends on $q$, so
$q=p_T$. Each step minimizes $G$ exactly over one argument, so $G$ cannot
increase, and it is bounded below by $\min\mathcal F$. That the iteration
converges to the global minimum uses the joint convexity of $G$ in
(encoder, $q$) (Csiszár and Tusnády 1984; quoted). This is the
Blahut–Arimoto algorithm (Blahut 1972; Arimoto 1972 for the channel-capacity
version; Cover and Thomas §10.8). In practice: put $t$ on a grid, start from a
uniform $q$, and iterate. `checks/wb2_new_ratedistortion_check.py` does this
and recovers $H_b(p)-H_b(D)$ to machine precision and
$\tfrac12\log(\sigma^2/D)$ on a grid to better than $10^{-3}$ relative error,
with $\beta=\log\frac{1-D}{D}$ and $\beta=\frac1{2D}$ respectively.

**9.** *Lower bound.* With independent components,
$h(X)=\sum_ih(X_i)$, and by the chain rule and "conditioning reduces entropy",
$h(X\mid T)=\sum_ih(X_i\mid T,X_1,\dots,X_{i-1})\le\sum_ih(X_i\mid T_i)$. So
$I(X;T)\ge\sum_iI(X_i;T_i)$. By part 4,
$I(X_i;T_i)\ge\tfrac12\log^+(\sigma_i^2/D_i)$ where
$D_i=\mathbb E(X_i-T_i)^2$ and $\log^+u=\max(\log u,0)$. Hence $R(D)$ is at
least the minimum of $\sum_i\tfrac12\log^+(\sigma_i^2/D_i)$ subject to
$\sum_iD_i\le D$.

*Allocation.* Taking $D_i>\sigma_i^2$ wastes distortion budget, so take
$D_i\le\sigma_i^2$. Minimize $\sum_i\tfrac12\log(\sigma_i^2/D_i)+\mu\big(\sum_iD_i-D\big)$:
for each $i$ with $D_i<\sigma_i^2$, $-\frac1{2D_i}+\mu=0$, so all such $D_i$
equal a common level $\theta=\frac1{2\mu}$. Components with $\sigma_i^2\le\theta$
sit at their bound $D_i=\sigma_i^2$. So $D_i=\min(\theta,\sigma_i^2)$.

*Achievability.* Use the part-5 test channel independently on each component
with $\sigma_i^2>\theta$, at distortion $\theta$, and set $T_i=0$ for the rest.
Then $I(X;T)=\sum_iI(X_i;T_i)$, and the bound is met.

*Picture.* Think of $\theta$ as a water level. Components whose variance is
below it are not described at all ($T_i=0$, error $\sigma_i^2$). Components
above it are all described to the same error $\theta$, at rate
$\tfrac12\log(\sigma_i^2/\theta)$, with forward encoder
$T_i=(1-\theta/\sigma_i^2)X_i+\text{noise}$. For example, with
$\sigma^2=(4,2,1,0.3)$ and $D=1.8$, the level is $\theta=0.5$, the fourth
component is dropped, and $R=\tfrac12\log(8\cdot4\cdot2)=3\log2$ nats.

*Correlated Gaussian.* Rotate to the eigenbasis, $X'=U^\top X$ with
$\Sigma=U\Lambda U^\top$, and use $T'=U^\top T$. Mutual information is
invariant under invertible maps of each variable, and squared error is
invariant under the same rotation of $x$ and $t$, so the problem becomes the
independent one with variances $\Lambda$. The optimal encoder keeps the
principal components with eigenvalue above $\theta$, shrinks each by
$1-\theta/\lambda_k$ and adds noise, and discards the rest: PCA truncation at a
level set by the distortion budget.

*Comparison with ridge.* Both are filters on principal directions that favour
high-variance directions, and both have a level below which directions are
suppressed: $n\lambda$ for ridge, $\theta$ here. Ridge's filter
$s_k^2/(s_k^2+n\lambda)=1/(1+n\lambda/s_k^2)$ is smooth and never exactly zero.
The rate–distortion filter $\max(0,1-\theta/\lambda_k)$ is exactly zero below
the level. Far above their levels the two agree to first order:
$1-n\lambda/s_k^2$ against $1-\theta/\lambda_k$. The
analogy is loose: ridge filters a regression coefficient using the input
covariance, while rate–distortion filters the source itself. What the two
share is that low-variance directions are where a fixed budget (of
information, or of signal-to-noise ratio) buys the least.
"""
    ),
]
