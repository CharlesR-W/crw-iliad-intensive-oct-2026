"""WB2 new material: a VC primer (2.C.1) and a short PAC-Bayes introduction (2.E).

Two separate pieces for section 2 ("Doesn't some theorem make this impossible?"):

- CELLS_VC: exercise group `#### 2.C.1. Shattering and VC dimension`, to sit at the start
  of 2.C, after the 2.C intro (written elsewhere) and before the old 2.C.i (now 2.C.2).
- CELLS_PACBAYES: subsection `### 2.E. PAC-Bayes: bounds for the function you found`,
  after 2.D (Zhang et al.).

Formulas are checked numerically by checks/wb2_new_pacbayes_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS_VC = [
    markdown(
        r"""
#### 2.C.1. Shattering and VC dimension

**Why.** The finite-class bound charges $\ln|\mathcal H|$, which is infinite
for any class with a real-valued parameter. What matters on $n$ training
points is not how many hypotheses there are but how many different
labellings of those points they produce. The VC dimension measures this, and
the parts below show it is not the number of parameters.

Let $\mathcal H$ be a class of binary classifiers $h:\mathcal X\to\{0,1\}$,
with risk $R(h)=\Pr_{(x,y)}[h(x)\ne y]$ and training error $\hat R_n(h)$ on
$n$ i.i.d. examples. A finite set $C=\{x_1,\dots,x_m\}\subset\mathcal X$ is
*shattered* by $\mathcal H$ if all $2^m$ labellings of $C$ are produced by
some $h\in\mathcal H$. The *growth function* $\Pi_{\mathcal H}(m)$ is the
largest number of distinct labellings that $\mathcal H$ produces on any $m$
points, so $\Pi_{\mathcal H}(m)\le2^m$, with equality iff some $m$-point set
is shattered. The *VC dimension* $d_{\mathrm{VC}}(\mathcal H)$ is the largest
$m$ for which some $m$-point set is shattered ($\infty$ if there is no
largest). To prove $d_{\mathrm{VC}}=d$, exhibit one shattered set of $d$
points and show that no set of $d+1$ points is shattered.

**Sauer–Shelah lemma** (quoted; proved in ch. 6 of Shalev-Shwartz and Ben-David's *Understanding Machine Learning*). If
$d_{\mathrm{VC}}(\mathcal H)=d<\infty$, then for $m\ge d$

$$
\Pi_{\mathcal H}(m)\le\sum_{i=0}^{d}\binom mi\le\Big(\frac{em}{d}\Big)^d .
$$

So on $n$ points a class of VC dimension $d$ acts like a finite class of
at most $(en/d)^d$ hypotheses, and $d\ln(en/d)$ takes the place of
$\ln|\mathcal H|$ in the finite-class bound. One standard form of the
resulting *VC bound* (quoted from Mohri, Rostamizadeh and Talwalkar,
*Foundations of Machine Learning*, 2nd ed., Corollary 3.19; it follows from
Rademacher complexity and Massart's lemma, and other textbooks have other
constants): with probability at least $1-\delta$, for all $h\in\mathcal H$,

$$
R(h)\le\hat R_n(h)+\sqrt{\frac{2d\ln(en/d)}{n}}+\sqrt{\frac{\ln(1/\delta)}{2n}} .
$$

It says something only when $n\gg d$. For ReLU networks with $W$ weights in
$L$ layers, the best general upper bound is $d=O(WL\log W)$, and some
architectures achieve $\Omega(WL\log(W/L))$
([Bartlett et al. 2019](https://jmlr.org/papers/v20/17-612.html)); at fixed
depth this is of order $W\log W$. 2.C.2 compares that with $n$.

1. *Thresholds* on $\mathbb R$: $h_t(x)=\mathbf 1[x\ge t]$, $t\in\mathbb R$. Show $d_{\mathrm{VC}}=1$ and $\Pi(m)=m+1$.
2. *Intervals* on $\mathbb R$: $h_{a,b}(x)=\mathbf 1[a\le x\le b]$. Show $d_{\mathrm{VC}}=2$, compute $\Pi(m)$, and compare with the Sauer–Shelah bound.
3. *Affine halfspaces* in $\mathbb R^d$: $h_{w,b}(x)=\mathbf 1[w^\top x+b\ge0]$, with $d+1$ parameters. Show that the $d+1$ points $0,e_1,\dots,e_d$ are shattered. Then use *Radon's theorem* (any $d+2$ points in $\mathbb R^d$ can be split into two groups whose convex hulls intersect) to show that no $d+2$ points are shattered, so $d_{\mathrm{VC}}=d+1$. Optionally, prove Radon's theorem.
4. *One parameter, infinite VC dimension.* Let $h_\omega(x)=\mathbf 1[\sin(\omega x)>0]$ with $\omega>0$. Take $x_i=2^{-i}$ for $i=1,\dots,m$, any labels $y_1,\dots,y_m\in\{0,1\}$, and $\omega=\pi\big(1+\sum_{j=1}^m(1-y_j)2^j\big)$. Show that $h_\omega(x_i)=y_i$ for all $i$, so $d_{\mathrm{VC}}=\infty$. What does this say about counting parameters? What changes if $\omega$ is stored as a 32-bit float?

<details><summary>Hint</summary>

For 2, sort the points: which patterns of 0s and 1s can an interval cut out?
For 3, the halfspace $\{x:w^\top x+b\ge0\}$ and its complement are both
convex. For Radon, solve $\sum_i\lambda_ix_i=0$, $\sum_i\lambda_i=0$ and
split the points by the sign of $\lambda_i$. For 4, compute
$\omega x_i/\pi$ modulo 2: the terms with $j>i$ are even integers.
</details>
"""
    ),
    answer(
        r"""
**1.** One point $x$ is shattered: $t\le x$ labels it 1 and $t>x$ labels it 0. For two points $x_1<x_2$, the labelling $(1,0)$ needs $t\le x_1$ and $t>x_2$, which is impossible. So $d_{\mathrm{VC}}=1$. On $m$ distinct sorted points the achievable labellings are $0\cdots01\cdots1$, fixed by the number $k\in\{0,\dots,m\}$ of zeros. So $\Pi(m)=m+1=\binom m0+\binom m1$, and Sauer–Shelah holds with equality.

**2.** Two points $x_1<x_2$: a long interval gives $(1,1)$, a short interval around one point gives $(1,0)$ or $(0,1)$, and an interval away from both gives $(0,0)$. Three points $x_1<x_2<x_3$: an interval containing $x_1$ and $x_3$ contains $x_2$, so $(1,0,1)$ is impossible, and $d_{\mathrm{VC}}=2$. On $m$ sorted points the 1s must form one contiguous block. There is the empty block, plus one block for each choice of first and last point $i\le j$, which gives

$$
\Pi(m)=1+\frac{m(m+1)}2=\binom m0+\binom m1+\binom m2 ,
$$

again equal to the Sauer–Shelah bound.

**3.** *Lower bound.* Write the labels of $0,e_1,\dots,e_d$ as $y_0,\dots,y_d$ and let $s_i=2y_i-1\in\{\pm1\}$. Take $b=s_0/2$ and $w_i=s_i-s_0/2$. Then $w^\top0+b=s_0/2$ and $w^\top e_i+b=s_i$. Each has the sign $s_i$ and none is zero, so every labelling is produced and $d_{\mathrm{VC}}\ge d+1$.

*Upper bound.* Take any $d+2$ points $x_1,\dots,x_{d+2}$. The system $\sum_i\lambda_ix_i=0$, $\sum_i\lambda_i=0$ has $d+1$ equations and $d+2$ unknowns, so it has a solution $\lambda\ne0$. Because the $\lambda_i$ sum to zero, some are positive and some negative. Let $I=\{i:\lambda_i>0\}$, let $J$ be the rest, and let $s=\sum_{i\in I}\lambda_i>0$. Then

$$
p=\sum_{i\in I}\frac{\lambda_i}{s}x_i=\sum_{i\in J}\frac{-\lambda_i}{s}x_i
$$

is a convex combination of the $I$-points and also of the $J$-points (Radon's theorem). Label $I$ by 1 and $J$ by 0. A halfspace producing this labelling would have the convex set $\{w^\top x+b\ge0\}$ contain all $I$-points, hence $p$. The convex set $\{w^\top x+b<0\}$ would contain all $J$-points, hence $p$ too. That is a contradiction. So $d_{\mathrm{VC}}=d+1$, which for this class equals the number of parameters.

**4.** Write $c_j=1-y_j\in\{0,1\}$. Then

$$
\frac{\omega x_i}{\pi}=2^{-i}\Big(1+\sum_{j=1}^mc_j2^j\Big)=\underbrace{\sum_{j>i}c_j2^{j-i}}_{\text{even integer}}+c_i+\theta_i,\qquad \theta_i=\sum_{j<i}c_j2^{j-i}+2^{-i}.
$$

The even integer shifts $\omega x_i$ by a multiple of $2\pi$ and does not change the sine. Also $0<\theta_i\le\sum_{k=1}^{i-1}2^{-k}+2^{-i}=1-2^{-i}<1$. So $\omega x_i\equiv\pi(c_i+\theta_i)\pmod{2\pi}$ with $\theta_i\in(0,1)$, and $\sin(\omega x_i)>0$ iff $c_i=0$ iff $y_i=1$. Every labelling of every finite set $\{2^{-1},\dots,2^{-m}\}$ is produced, so $d_{\mathrm{VC}}=\infty$. No distribution-free VC bound applies to this one-parameter class.

The number of parameters is neither an upper bound on the VC dimension (this example; ReLU networks with $W$ weights in $L$ layers, whose VC dimension can be of order $WL\log(W/L)$) nor a lower bound (redundant parameters, as in $\mathbf 1[(w_1+w_2)x\ge0]$, add nothing).

If $\omega$ is a 32-bit float, the class has at most $2^{32}$ members. Shattering $m$ points needs $2^m$ distinct functions, so $d_{\mathrm{VC}}\le32$, and the finite-class bound applies with $\ln|\mathcal H|\le32\ln2$. The construction above needs $\omega/\pi$ to about $m+1$ significant bits. The infinite VC dimension comes from the infinite precision of one real number, which can hide arbitrarily many bits. This is the difference between counting parameters and counting bits that the float-encoding bound in 2.C.2 relies on.
"""
    ),
]

CELLS_PACBAYES = [
    markdown(
        r"""
### 2.E. PAC-Bayes: bounds for the function you found

**The question.** 2.C and 2.D showed that any generalization bound depending
only on the hypothesis class, the sample size and the confidence level must be
vacuous for networks that can fit random labels. Is there a rigorous bound
that can still be small for the network SGD actually found? PAC-Bayes bounds
are one answer. Their complexity term measures how far a distribution over
weights, built around the trained network after seeing the data, is from a
distribution fixed before seeing it. The bound is small only if a distribution
close to the prior, and hence broad, still has low training error. It depends
on the data and on the solution, not on what else the class could fit. With
such a bound, Dziugaite and Roy (2017, reading below) obtained the first
nonvacuous bounds for networks with millions of parameters, where VC bounds
were vacuous by orders of magnitude. In this subsection you prove a PAC-Bayes
bound from scratch and recover the Occam bound as a special case (2.E.1), then
see that for Gaussian distributions over weights the complexity term becomes
a weight-norm term plus a flatness term, and that the bound is consistent with
Zhang et al.'s random-label fits (2.E.2).

**Setup.** Examples $z=(x,y)$ are drawn i.i.d. from an unknown distribution
$D$, and $S=(z_1,\dots,z_n)$ is the sample. Hypotheses are $h\in\mathcal H$;
for a network, $h$ is a weight vector $w\in\mathbb R^p$. The loss
$\ell(h;z)$ takes values in $[0,1]$, for example the 0–1 loss. The risk is
$R(h)=\mathbb E_{z\sim D}\,\ell(h;z)$ and the empirical risk is
$\hat R_n(h)=\frac1n\sum_{i=1}^n\ell(h;z_i)$.

- The *prior* $P$ is a probability distribution on $\mathcal H$ chosen without looking at $S$. It may depend on anything independent of the sample, such as the random initialization.
- The *posterior* $Q$ is any probability distribution on $\mathcal H$, and may depend on $S$ in any way. Despite the names, there is no likelihood and no Bayes rule. A typical choice is $\mathcal N(w_{\mathrm{SGD}},\sigma^2I)$, centred on the trained weights.
- The *Gibbs predictor* draws a fresh $h\sim Q$ for each prediction. Its risk is $\mathbb E_{h\sim Q}R(h)$, written $\mathbb E_QR$, and its training risk is $\mathbb E_Q\hat R_n$.
- The *Kullback–Leibler divergence* is $\mathrm{KL}(Q\|P)=\mathbb E_{h\sim Q}\ln\frac{dQ}{dP}(h)\ge0$: $\sum_hQ(h)\ln\frac{Q(h)}{P(h)}$ for discrete distributions and $\int q\ln\frac qp$ for densities. It is $+\infty$ unless $Q$ gives zero mass to every set to which $P$ gives zero mass.

**Theorem (McAllester's bound with Maurer's constants; quoted from Alquier's
tutorial, §3.2.2).** Fix the prior $P$ and $\delta\in(0,1)$. With probability
at least $1-\delta$ over the sample $S$, simultaneously for all $Q$,

$$
\mathbb E_QR\le\mathbb E_Q\hat R_n+\sqrt{\frac{\mathrm{KL}(Q\|P)+\ln(2\sqrt n/\delta)}{2n}} .
$$

- *For all $Q$ simultaneously.* You may choose $Q$ after seeing the data, for example by minimizing the right-hand side. Only $P$ must be fixed in advance.
- *What replaced the class.* The one-sided finite-class bound, uniform over a class of size $|\mathcal H|$, reads $R(h)\le\hat R_n(h)+\sqrt{(\ln|\mathcal H|+\ln(1/\delta))/(2n)}$. Here $\ln|\mathcal H|$, a property of the class, has become $\mathrm{KL}(Q\|P)$, a property of what you chose after seeing the data. 2.E.1.5 makes this exact.
- *Which predictor.* The bound is for the randomized predictor, not for the single network at the centre of $Q$. Transferring it to that network needs a further argument, for example that its predictions rarely change under the perturbations $Q$ makes. Nothing requires training to sample from $Q$, or from any Bayesian posterior.

Numerical certificates use the sharper form the theorem is derived from
(Alquier, Theorem 3.2):
$\mathrm{kl}\big(\mathbb E_Q\hat R_n\,\|\,\mathbb E_QR\big)\le\big(\mathrm{KL}(Q\|P)+\ln(2\sqrt n/\delta)\big)/n$,
where $\mathrm{kl}(q\|r)=q\ln\frac qr+(1-q)\ln\frac{1-q}{1-r}$ is the KL
divergence between two coin flips. Pinsker's inequality
$\mathrm{kl}(q\|r)\ge2(q-r)^2$ gives the displayed theorem; near zero training
error the kl form decays like $1/n$ rather than $1/\sqrt n$.

#### 2.E.1. Change of measure, and a PAC-Bayes bound in five steps

The standard PAC-Bayes proofs rest on one inequality, the Donsker–Varadhan (or Gibbs)
variational formula of part 1. Parts 2–3 use it to prove a bound due to
Catoni (Alquier, Theorem 2.1): for any $\lambda>0$ fixed before seeing the
data, with probability at least $1-\delta$, simultaneously for all $Q$,

$$
\mathbb E_QR\le\mathbb E_Q\hat R_n+\frac{\mathrm{KL}(Q\|P)+\ln(1/\delta)}{\lambda}+\frac{\lambda}{8n} .\qquad(\star)
$$

1. *Donsker–Varadhan.* Let $f$ be a bounded function on $\mathcal H$, and define the Gibbs measure $P_f$ by $\frac{dP_f}{dP}=\frac{e^f}{\mathbb E_Pe^f}$. Show that for every $Q$, $\mathbb E_Qf\le\mathrm{KL}(Q\|P)+\ln\mathbb E_Pe^f$, with equality iff $Q=P_f$. Equivalently, $\ln\mathbb E_Pe^f=\max_Q\big[\mathbb E_Qf-\mathrm{KL}(Q\|P)\big]$. What is this for $f=-\beta E$ and $P$ uniform on finitely many states?
2. *One hypothesis.* Fix $h$ and $\lambda>0$. Use Hoeffding's lemma (if $X\in[a,b]$ and $\mathbb EX=0$, then $\mathbb Ee^{sX}\le e^{s^2(b-a)^2/8}$) to show $\mathbb E_S\,e^{\lambda(R(h)-\hat R_n(h))}\le e^{\lambda^2/(8n)}$.
3. *All posteriors at once.* Show that with probability at least $1-\delta$ over $S$, $\mathbb E_{h\sim P}\,e^{\lambda(R(h)-\hat R_n(h))}\le e^{\lambda^2/(8n)}/\delta$. Where do you use that $P$ does not depend on $S$? On this event, apply part 1 with $f=\lambda(R-\hat R_n)$ to obtain $(\star)$ for all $Q$.
4. *Choosing $\lambda$, and the best $Q$.* For fixed $Q$, minimize the right-hand side of $(\star)$ over $\lambda$ and compare with the theorem above. Why can you not simply plug in the minimizing $\lambda$, and how does a union bound over a grid of $\lambda$ values fixed in advance repair it? Then, for fixed $\lambda$, show that the $Q$ minimizing the right-hand side of $(\star)$ is the Gibbs posterior $Q_\lambda$ with $\frac{dQ_\lambda}{dP}\propto e^{-\lambda\hat R_n}$. When would $Q_\lambda$ be a Bayesian posterior?
5. *Occam.* Let $\mathcal H$ be countable and take $Q=\delta_h$, the point mass on one hypothesis. Show $\mathrm{KL}(\delta_h\|P)=\ln(1/P(h))$, and write the bound the theorem gives, which holds for all $h$ simultaneously. What does it give for $P$ uniform on a finite class, and for $P(h)\propto2^{-L(h)}$, where $L(h)$ is the length in bits of $h$ in a prefix-free code? Why is a point mass useless for real-valued weights under a prior with a density?

<details><summary>Hint</summary>

For 1, expand $\mathrm{KL}(Q\|P_f)\ge0$. For 2, write
$R(h)-\hat R_n(h)=\frac1n\sum_iX_i$ with $X_i=R(h)-\ell(h;z_i)$: independent,
mean zero, each in an interval of length 1. Take $s=\lambda/n$. For 3, average
part 2 over $h\sim P$, swap $\mathbb E_S$ and $\mathbb E_P$, and apply
Markov's inequality to the nonnegative random variable
$\mathbb E_Pe^{\lambda(R-\hat R_n)}$. For 4, use the grid $\lambda_k=2^k$,
$k=0,\dots,K-1$, with confidence $\delta/K$ for each. For 5, Kraft's
inequality says that the code lengths of a prefix-free binary code satisfy
$\sum_h2^{-L(h)}\le1$.
</details>
"""
    ),
    answer(
        r"""
**1.** If $\mathrm{KL}(Q\|P)=\infty$ there is nothing to prove. Otherwise write $Z=\mathbb E_Pe^f$, so $\frac{dQ}{dP_f}=\frac{dQ}{dP}\,Ze^{-f}$ and

$$
0\le\mathrm{KL}(Q\|P_f)=\mathbb E_Q\Big[\ln\frac{dQ}{dP}-f+\ln Z\Big]=\mathrm{KL}(Q\|P)-\mathbb E_Qf+\ln\mathbb E_Pe^f .
$$

Rearranging gives the inequality, and $\mathrm{KL}(Q\|P_f)=0$ iff $Q=P_f$, so the maximum is attained exactly at the Gibbs measure. With $P$ uniform on $N$ states, $\mathrm{KL}(Q\|P)=\ln N-\mathrm{Ent}(Q)$, where $\mathrm{Ent}(Q)=-\sum Q\ln Q$ is the entropy. Taking $f=-\beta E$ gives $\mathbb E_Pe^f=Z_\beta/N$, with $Z_\beta=\sum e^{-\beta E}$ the partition function. The inequality becomes $\langle E\rangle_Q-T\,\mathrm{Ent}(Q)\ge-T\ln Z_\beta=F$, with equality for the Boltzmann distribution: the Gibbs variational principle for the free energy.

**2.** The $X_i=R(h)-\ell(h;z_i)$ are i.i.d., have mean zero, and lie in $[R(h)-1,R(h)]$, an interval of length 1. By independence and Hoeffding's lemma with $s=\lambda/n$,

$$
\mathbb E_S\,e^{\lambda(R(h)-\hat R_n(h))}=\prod_{i=1}^n\mathbb E\,e^{(\lambda/n)X_i}\le\big(e^{\lambda^2/(8n^2)}\big)^n=e^{\lambda^2/(8n)} .
$$

**3.** Average part 2 over $h\sim P$. Since $P$ is fixed and the integrand is nonnegative, Tonelli's theorem lets you swap the order: $\mathbb E_S\big[\mathbb E_{h\sim P}e^{\lambda(R-\hat R_n)}\big]\le e^{\lambda^2/(8n)}$. This is where $P$ must not depend on $S$. If it did, the inner average would be over a different distribution for each sample, and the swap would no longer reduce to part 2. Markov's inequality applied to $Y=\mathbb E_Pe^{\lambda(R-\hat R_n)}\ge0$ gives $\Pr_S\big[Y>e^{\lambda^2/(8n)}/\delta\big]\le\delta$. On the complementary event, part 1 with $f=\lambda(R-\hat R_n)$ (bounded, since the loss is) gives, for every $Q$,

$$
\lambda\,\mathbb E_Q(R-\hat R_n)\le\mathrm{KL}(Q\|P)+\ln Y\le\mathrm{KL}(Q\|P)+\frac{\lambda^2}{8n}+\ln\frac1\delta .
$$

Dividing by $\lambda$ gives $(\star)$. The event was defined without reference to $Q$, which is why the bound holds for all $Q$ at once, including $Q$ chosen using $S$.

**4.** Let $a=\mathrm{KL}(Q\|P)+\ln(1/\delta)$. Then $a/\lambda+\lambda/(8n)$ is minimized at $\lambda^*=\sqrt{8na}$, with value $\sqrt{a/(2n)}$. This would give $\mathbb E_QR\le\mathbb E_Q\hat R_n+\sqrt{(\mathrm{KL}+\ln(1/\delta))/(2n)}$, the theorem's form with $\ln(1/\delta)$ in place of $\ln(2\sqrt n/\delta)$. But $\lambda^*$ depends on $\mathrm{KL}(Q\|P)$, and $Q$ depends on $S$. The event in part 3 was built for a single $\lambda$ chosen in advance, and nothing covers a $\lambda$ chosen after seeing the data.

*Repair.* Before seeing the data, fix the grid $\lambda_k=2^k$, $k=0,\dots,K-1$, with $2^{K-1}\ge4n$, so $K\approx\log_2(4n)+1$. Apply part 3 to each $\lambda_k$ at confidence $\delta/K$. By the union bound, with probability at least $1-\delta$ all $K$ bounds hold at once, for all $Q$, with $\ln(K/\delta)$ in place of $\ln(1/\delta)$, so you may pick the best $k$ after seeing the data. With $a=\mathrm{KL}(Q\|P)+\ln(K/\delta)$ and whenever the optimized bound is below 1, the optimal $\lambda^*=\sqrt{8na}$ lies in $[1,4n]$, hence within a factor $\sqrt2$ of a grid point; at $\lambda=c\lambda^*$ the value is $\frac12(c+1/c)\sqrt{a/(2n)}$, at most $1.061\sqrt{a/(2n)}$ for $c\in[1/\sqrt2,\sqrt2]$. The cost is that factor and $\ln K\sim\ln\ln n$ extra nats; the theorem pays $\ln(2\sqrt n)$ instead.

*Best $Q$.* Only $\mathbb E_Q\hat R_n+\mathrm{KL}(Q\|P)/\lambda$ depends on $Q$, and by part 1 with $f=-\lambda\hat R_n$,

$$
\min_Q\Big[\mathbb E_Q\hat R_n+\frac{\mathrm{KL}(Q\|P)}\lambda\Big]=-\frac1\lambda\ln\mathbb E_Pe^{-\lambda\hat R_n},
$$

attained at $Q_\lambda\propto e^{-\lambda\hat R_n}P$. If the loss were the negative log-likelihood $-\ln p(y\mid x,h)$ and $\lambda=n$, then $e^{-n\hat R_n}=\prod_ip(y_i\mid x_i,h)$, and $Q_\lambda$ would be exactly the Bayesian posterior. Other values of $\lambda$ give "tempered" posteriors. The log loss is unbounded, so the $[0,1]$ theorem does not cover that case as stated. The Bayesian posterior is one possible $Q$ among many, and the bound certifies whichever $Q$ you choose.

**5.** $\mathrm{KL}(\delta_h\|P)=1\cdot\ln\frac1{P(h)}$, since $\delta_h$ puts all its mass on $h$. With probability at least $1-\delta$, for all $h$ simultaneously,

$$
R(h)\le\hat R_n(h)+\sqrt{\frac{\ln(1/P(h))+\ln(2\sqrt n/\delta)}{2n}} .
$$

- *Uniform $P$ on $M$ hypotheses:* $\ln(1/P(h))=\ln M$. This is the finite-class bound, except for $\ln(2\sqrt n/\delta)$ in place of $\ln(1/\delta)$; a direct union bound avoids that extra $\ln(2\sqrt n)$.
- *Prefix code:* Kraft's inequality $\sum_h2^{-L(h)}\le1$ makes $P(h)=2^{-L(h)}/Z$ a distribution with $Z\le1$, so $\ln(1/P(h))\le L(h)\ln2$. The complexity is the description length of the particular $h$ you output, $L(h)\ln2$ nats, regardless of how large the class is. This is the Occam bound: hypotheses with short descriptions need few samples.
- *Continuous weights:* if $P$ has a density, then $P(\{w\})=0$ and $\mathrm{KL}(\delta_w\|P)=\infty$, so the bound is vacuous. To pay a finite price for a real-valued $w$, either store it to finite precision (a code, as above) or spread $Q$ over a neighbourhood of $w$ whose prior mass is not too small. 2.E.2 does the second.
"""
    ),
    markdown(
        r"""
#### 2.E.2. Gaussian posteriors: weight norm and flatness

Take network weights $w\in\mathbb R^p$, the prior
$P=\mathcal N(w_0,\sigma_P^2I)$, with $w_0$ and $\sigma_P$ fixed before
seeing the data (for example $w_0=0$, or the random initialization), and the
posterior $Q=\mathcal N(w,\Sigma_Q)$ centred on the trained weights $w$. Use
Catoni's bound $(\star)$ from 2.E.1 with $\lambda$ fixed in advance: with
probability at least $1-\delta$, for all $Q$,
$\mathbb E_QR\le\mathbb E_Q\hat R_n+\big(\mathrm{KL}(Q\|P)+\ln(1/\delta)\big)/\lambda+\lambda/(8n)$.
For parts 2–4, treat $\hat R_n$ as smooth near $w$, with Hessian
$\hat H=\nabla^2\hat R_n(w)$. This holds for a bounded smooth surrogate loss;
for the 0–1 loss these parts are heuristic.

1. Show that for Gaussians in $\mathbb R^p$, $\mathrm{KL}\big(\mathcal N(\mu_Q,\Sigma_Q)\,\|\,\mathcal N(\mu_P,\Sigma_P)\big)=\tfrac12\big[\mathrm{tr}(\Sigma_P^{-1}\Sigma_Q)+(\mu_Q-\mu_P)^\top\Sigma_P^{-1}(\mu_Q-\mu_P)-p+\ln\frac{\det\Sigma_P}{\det\Sigma_Q}\big]$. Deduce that, with $s_1,\dots,s_p$ the eigenvalues of $\Sigma_Q$, $\mathrm{KL}(Q\|P)=\frac{\|w-w_0\|^2}{2\sigma_P^2}+\frac12\sum_{k=1}^p\Big(\frac{s_k}{\sigma_P^2}-1-\ln\frac{s_k}{\sigma_P^2}\Big)$. Interpret the two terms. What does the second cost for $\Sigma_Q=\sigma_Q^2I$ with $\sigma_Q\ll\sigma_P$, and how big is that for $p=10^6$ and $\sigma_P/\sigma_Q=10^3$?
2. Show that $\mathbb E_Q\hat R_n=\hat R_n(w)+\tfrac12\mathrm{tr}(\hat H\Sigma_Q)$ up to fourth-order terms in the width of $Q$, whether or not $\nabla\hat R_n(w)=0$.
3. Insert parts 1 and 2 into $(\star)$ and minimize over $\Sigma_Q$, assuming $\hat H\succeq0$ and that the quadratic approximation holds over the width of $Q$. Show that the optimum is $\Sigma_Q=(\sigma_P^{-2}I+\lambda\hat H)^{-1}$ and that the bound becomes $\mathbb E_QR\lesssim\hat R_n(w)+\frac1\lambda\Big[\frac{\|w-w_0\|^2}{2\sigma_P^2}+\frac12\ln\det\big(I+\lambda\sigma_P^2\hat H\big)+\ln\frac1\delta\Big]+\frac\lambda{8n}$. Which eigendirections of $\hat H$ cost complexity, and how much? Where does weight decay appear?
4. *The prior picks a metric (Dinh et al.'s rescaling).* Take the two-layer scalar linear network $f(x)=w_2w_1x$ with $\hat R_n(w)=a(w_1w_2-c)^2+\text{const}$, $a>0$, $c\ne0$: the example of 1.D.7.3, with the training risk in place of the population risk. Its minima form the hyperbola $w_1w_2=c$; at a minimum $\hat H=2a\,vv^\top$ with $v=(w_2,w_1)^\top$. The rescaling $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ does not change the function; Dinh et al. (1.D.8) use this symmetry to make ReLU minima arbitrarily sharp. (This is illustrative only: the squared loss needs a bounded surrogate, and the quadratic approximation needs $\sigma_P\ll|w|$.) With $w_0=0$, show that the norm term and the log-det term of part 3 depend on $\alpha$ only through $\rho(\alpha)=\alpha^2w_1^2+w_2^2/\alpha^2$, so the bound is smallest at the balanced representative and unbounded as $\alpha\to0$ or $\infty$. Is the bound then wrong, or only loose? Show that $\mathrm{KL}(Q\|P)$ is unchanged when the same smooth bijection of weight space is applied to both $P$ and $Q$. Where did the dependence on the representative come from?
5. *Random labels.* Replace the labels by labels drawn uniformly from 10 classes, independently of $x$, and use the 0–1 loss. Using the theorem at the start of 2.E, show that with probability at least $1-\delta$, every $Q$ with $\mathbb E_Q\hat R_n\le0.1$ has $\mathrm{KL}(Q\|P)\ge2n(0.8)^2-\ln(2\sqrt n/\delta)$. Evaluate this for $n=5\times10^4$ and $\delta=0.05$. Why is this consistent with SGD fitting random labels? Why does it not run into the argument of 2.C.2.3, that every bound depending only on the class, $n$ and $\delta$ must be at least $0.9$?

<details><summary>Hint</summary>

For 1, take the expectation under $Q$ of $\ln q-\ln p$, using
$\mathbb E_Q(x-m)^\top A(x-m)=\mathrm{tr}(A\Sigma_Q)+(\mu_Q-m)^\top A(\mu_Q-m)$.
For 2, expand $\hat R_n(w+\varepsilon)$ with $\varepsilon\sim\mathcal N(0,\Sigma_Q)$;
odd moments of $\varepsilon$ vanish. For 3, use
$\partial_\Sigma\,\mathrm{tr}(A\Sigma)=A$ and $\partial_\Sigma\ln\det\Sigma=\Sigma^{-1}$,
and simplify with $\mathrm{tr}\big(\Sigma_Q(\lambda\hat H+\sigma_P^{-2}I)\big)=p$.
For 5, what is $R(h)$ for any fixed $h$?
</details>
"""
    ),
    answer(
        r"""
**1.** With $\Delta=\mu_Q-\mu_P$,

$$
\ln\frac{q(x)}{p(x)}=\frac12\ln\frac{\det\Sigma_P}{\det\Sigma_Q}-\frac12(x-\mu_Q)^\top\Sigma_Q^{-1}(x-\mu_Q)+\frac12(x-\mu_P)^\top\Sigma_P^{-1}(x-\mu_P).
$$

Under $Q$, the first quadratic form has mean $\mathrm{tr}(\Sigma_Q^{-1}\Sigma_Q)=p$, and the second has mean $\mathrm{tr}(\Sigma_P^{-1}\Sigma_Q)+\Delta^\top\Sigma_P^{-1}\Delta$, which gives the formula. With $\Sigma_P=\sigma_P^2I$, $\mathrm{tr}(\Sigma_P^{-1}\Sigma_Q)=\sum_ks_k/\sigma_P^2$ and $\ln\det\Sigma_P/\det\Sigma_Q=\sum_k\ln(\sigma_P^2/s_k)$.

- *Norm term.* $\|w-w_0\|^2/(2\sigma_P^2)$ is an $L_2$ penalty on the distance travelled from $w_0$ (see part 3).
- *Width term.* Each eigendirection costs $\frac12\phi(s_k/\sigma_P^2)$, with $\phi(t)=t-1-\ln t\ge0$, which is zero iff $t=1$. Narrowing $Q$ below the prior width costs complexity. For $\Sigma_Q=\sigma_Q^2I$ with $\sigma_Q\ll\sigma_P$, the cost is about $p\ln(\sigma_P/\sigma_Q)$: roughly the nats needed to specify each of $p$ coordinates to precision $\sigma_Q$ within a range $\sigma_P$. For $p=10^6$ and $\sigma_P/\sigma_Q=10^3$, it is $\frac p2(10^{-6}-1+\ln10^6)\approx6.4\times10^6$ nats, hopeless for $n=5\times10^4$.

So a nonvacuous bound needs $Q$ to be about as wide as the prior in almost every direction. In Dziugaite and Roy's runs, $\mathrm{KL}\approx5\times10^3$ nats for about $5\times10^5$ parameters, roughly $0.01$ nats per parameter.

**2.** Write $\varepsilon=w'-w\sim\mathcal N(0,\Sigma_Q)$. Then $\hat R_n(w+\varepsilon)=\hat R_n(w)+\nabla\hat R_n(w)^\top\varepsilon+\frac12\varepsilon^\top\hat H\varepsilon+O(\|\varepsilon\|^3)$. The linear term has mean zero, whether or not the gradient vanishes, and $\mathbb E\,\varepsilon^\top\hat H\varepsilon=\mathrm{tr}(\hat H\Sigma_Q)$. The cubic term also has mean zero (odd Gaussian moments), so the error is fourth order, and the formula is exact for a quadratic $\hat R_n$. The randomized predictor pays for curvature in proportion to how widely it spreads. This is where flatness enters.

**3.** The $\Sigma$-dependent part of the bound is

$$
J(\Sigma)=\frac12\mathrm{tr}(\hat H\Sigma)+\frac1{2\lambda}\Big[\frac{\mathrm{tr}\,\Sigma}{\sigma_P^2}-p-\ln\det\Sigma+p\ln\sigma_P^2\Big].
$$

It is convex, since $-\ln\det$ is convex. Setting the gradient to zero gives $\frac12\hat H+\frac1{2\lambda}(\sigma_P^{-2}I-\Sigma^{-1})=0$, so $\Sigma_Q^{-1}=\sigma_P^{-2}I+\lambda\hat H$. At the optimum, $\mathrm{tr}(\hat H\Sigma_Q)+\mathrm{tr}(\Sigma_Q)/(\lambda\sigma_P^2)=\frac1\lambda\mathrm{tr}\big(\Sigma_Q(\lambda\hat H+\sigma_P^{-2}I)\big)=p/\lambda$, so

$$
J=\frac p{2\lambda}-\frac p{2\lambda}+\frac1{2\lambda}\ln\det\big(\sigma_P^2\Sigma_Q^{-1}\big)=\frac1{2\lambda}\ln\det\big(I+\lambda\sigma_P^2\hat H\big).
$$

Adding the norm term, $\ln(1/\delta)/\lambda$ and $\lambda/(8n)$ gives the stated bound.

In the eigenbasis of $\hat H$, with eigenvalues $h_k\ge0$, the posterior variance is $s_k=\sigma_P^2/(1+\lambda\sigma_P^2h_k)$, and the direction contributes $\frac1{2\lambda}\ln(1+\lambda\sigma_P^2h_k)$ to the bound (its share of the KL plus its curvature penalty, times $1/\lambda$).
- *Flat directions*, $h_k\ll1/(\lambda\sigma_P^2)$: the posterior is as wide as the prior and the contribution is about zero, however many such directions there are.
- *Sharp directions*: the posterior width is about $1/\sqrt{\lambda h_k}$ and the contribution about $\frac1{2\lambda}\ln(\lambda\sigma_P^2h_k)$, only logarithmic in the sharpness.

The bound counts the curved directions, not the parameters. This is the PAC-Bayes form of the old minimum-description-length argument that flat minima need fewer bits. Weight decay is the norm term, with $w$ held fixed here. Optimizing $w$ as well trades $\hat R_n(w)$ against $\|w-w_0\|^2/(2\lambda\sigma_P^2)$, which is weight decay towards $w_0$. Neither $\sigma_P$ nor $\lambda$ may be tuned on the data for free; both need a grid and a union bound, as in 2.E.1.4. (The Laplace approximation to the Bayesian evidence has the same log-determinant, with $n$ in place of $\lambda$.)

**4.** Write $\rho(\alpha)=\alpha^2w_1^2+w_2^2/\alpha^2$. The norm term is $\|w\|^2/(2\sigma_P^2)=\rho(\alpha)/(2\sigma_P^2)$. At the rescaled point $\hat H=2a\,v_\alpha v_\alpha^\top$ with $v_\alpha=(w_2/\alpha,\alpha w_1)^\top$. This has rank one with nonzero eigenvalue $2a\|v_\alpha\|^2=2a\rho(\alpha)$, so $\ln\det(I+\lambda\sigma_P^2\hat H)=\ln\big(1+2a\lambda\sigma_P^2\rho(\alpha)\big)$. Both terms increase with $\rho$. By AM–GM, $\rho\ge2|w_1w_2|=2|c|$, with equality iff $\alpha^2=|w_2/w_1|$, the balanced point, and $\rho\to\infty$ as $\alpha\to0$ or $\infty$.

*Wrong or loose?* Every one of these bounds is true, because the theorem holds for all $Q$ at once. They are different numbers for the same function. Since $Q$ may be chosen after seeing the data, you are free to centre it at the balanced representative, so the rescaling cannot make the best available bound worse. Dinh et al.'s construction (sharp minima, same function) shows only that the representative SGD lands on need not be the one that gives the best bound.

*Invariance.* If $\varphi$ is a smooth bijection and $P'=\varphi_\#P$, $Q'=\varphi_\#Q$, both densities pick up the same Jacobian factor, which cancels in $q'/p'$, so $\mathrm{KL}(Q'\|P')=\mathrm{KL}(Q\|P)$. The dependence on the representative came from choosing isotropic Gaussians *in the $w$ coordinates*: an isotropic Gaussian in rescaled coordinates is a different prior, which favours different representatives. Choosing $P$ is choosing a metric, and so a notion of "flat", which is the requirement 1.D.7.4 found for any meaningful flatness claim. Because the loss depends on $w$ only through the function $f_w$, the theorem applies equally to distributions over functions, and by the data-processing inequality their KL is no larger than the weight-space KL. Weight-space KL overcharges by paying to specify which of many equivalent $w$ you hold (Dziugaite and Roy, Appendix B).

**5.** For any fixed $h$, $\Pr[h(x)\ne y]=0.9$, because $y$ is uniform on 10 classes and independent of $x$, hence of $h(x)$. So $\mathbb E_QR=0.9$ for every $Q$. On the theorem's event, $0.9\le\mathbb E_Q\hat R_n+\sqrt{(\mathrm{KL}+\ln(2\sqrt n/\delta))/(2n)}$. If $\mathbb E_Q\hat R_n\le0.1$, then $\mathrm{KL}(Q\|P)\ge2n(0.8)^2-\ln(2\sqrt n/\delta)$. For $n=5\times10^4$ and $\delta=0.05$, $\ln(2\sqrt n/\delta)\approx9.1$, so $\mathrm{KL}\ge6.4\times10^4$ nats, about $1.3$ nats per training example.

This is consistent with fitting random labels. SGD finds $w$ with $\hat R_n(w)=0$, but every $Q$ that keeps the Gibbs training error below $0.1$ must be far from $P$: narrow around a sharp memorizing solution, or far from $w_0$, or both. The theorem predicts its own vacuity on such data. On real labels nothing forbids a small KL. Dziugaite and Roy found $Q$ with $\mathrm{KL}\approx5\text{–}9\times10^3$ nats at $n=55{,}000$, and nonvacuous bounds. Their random-label run (two classes) ended at $\mathrm{KL}\approx2\times10^5$, with a bound of $1.35$.

The argument of 2.C.2.3 applies to bounds of the form $\epsilon(n,\delta)$ that depend only on the class, $n$ and $\delta$. Such a bound must also hold on the random-label distribution, so it is at least $0.9$. The PAC-Bayes right-hand side also depends on $S$ through $Q$ and $\mathbb E_Q\hat R_n$. It can be large on random labels and small on real ones, with the same prior, architecture and $n$. What the bound does not explain is *why* SGD on real data lands near wide, low-KL regions. It certifies that when it happens; it does not predict it.
"""
    ),
    markdown(
        r"""
**Readings.**
"""
        + reading(
            "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data",
            "https://arxiv.org/abs/1703.11008",
            "Gintare Karolina Dziugaite, Daniel M. Roy (UAI 2017)",
            "Starting from SGD solutions on a two-class version of MNIST (55,000 training images), the authors "
            "optimize a diagonal Gaussian posterior around the trained weights by gradient descent on a PAC-Bayes "
            "bound. The prior is centred at the random initialization, and its variance is chosen from a grid via a "
            "union bound. For networks with 0.3–2.4 million parameters, the bounds on the error of the stochastic "
            "networks are 0.16–0.22, against a measured test error of about 0.03 and VC-dimension bounds that are "
            "vacuous by orders of magnitude. On random labels the bound is vacuous.",
            "Read §1 (the motivation from Zhang et al. and from flat minima), §2–3 (the kl-form bound, the choice "
            "of prior, and the objective: a McAllester-type bound for a diagonal Gaussian posterior, whose KL term "
            "is the one in 2.E.2.1) and §5 (results, Table 1). "
            "Appendix B, on how network symmetries loosen the bound, is optional.",
            importance=5, length=3,
        )
        + references(
            book(
                "User-friendly introduction to PAC-Bayes bounds",
                "https://arxiv.org/abs/2110.11216",
                "Pierre Alquier (Foundations and Trends in Machine Learning, 2024)",
                "§2.1 (Catoni's bound), §3.2–3.3",
            ),
        )
    ),
]

CELLS = CELLS_VC + CELLS_PACBAYES
