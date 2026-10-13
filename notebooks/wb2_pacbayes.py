"""MDL1 §6: PAC-Bayes, bounds for the function you found.

Formulas are checked by checks/wb2_new_pacbayes_check.py.
"""
from nbkit import *  # noqa: F401,F403

CELLS = [
    markdown(
        r"""
<a id="wb2-section-6"></a>

## 6. PAC-Bayes: bounds for the function you found

A rigorous bound that can be small for the network SGD actually found, because
its complexity term depends on the data and the solution rather than on what
else the class could fit; it is the one classical bound that survives the
keystone's argument.

Notation: $P$ is the prior, $Q$ the posterior, $\mathrm{KL}(Q\|P)$ their divergence, $\delta$ the confidence level; $\lambda>0$ is Catoni's free parameter (Alquier's notation), not a ridge penalty; $\sigma_P^2$ is the prior variance, $w$ the trained weights, $\hat H$ the training-loss Hessian.
"""
        + reading(
            "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data",
            "https://arxiv.org/abs/1703.11008",
            "Gintare Karolina Dziugaite, Daniel M. Roy (UAI 2017)",
            "Optimizing a Gaussian posterior around SGD solutions on binary MNIST gives error bounds of "
            "0.16–0.22 for networks with up to 2.4 million parameters, where VC bounds are vacuous by orders of "
            "magnitude; on random labels the bound is vacuous. Read §1–3 and §5.",
            importance=5, length=3,
        )
        + references(
            book("User-friendly introduction to PAC-Bayes bounds", "https://arxiv.org/abs/2110.11216",
                 "Pierre Alquier (Foundations and Trends in Machine Learning, 2024)",
                 "§2.1 (Catoni's bound), §3.2–3.3"),
        )
        + r"""
### Exercises: a PAC-Bayes bound from scratch

**Setup for 6.1–6.2.** Examples $z$ are drawn i.i.d. from $D$; $S=(z_1,\dots,z_n)$.
Hypotheses $h\in\mathcal H$ (for a network, a weight vector $w\in\mathbb R^p$)
have a loss $\ell(h;z)\in[0,1]$, risk $R(h)=\mathbb E_{z\sim D}\ell(h;z)$ and
empirical risk $\hat R_n(h)$.

- The *prior* $P$ is a distribution on $\mathcal H$ chosen without looking at $S$ (it may depend on the random initialization).
- The *posterior* $Q$ is any distribution on $\mathcal H$, and may depend on $S$ in any way; there is no likelihood and no Bayes rule. A typical choice is $\mathcal N(w_{\mathrm{SGD}},\sigma^2I)$.
- The *Gibbs predictor* draws a fresh $h\sim Q$ for each prediction; its risk is $\mathbb E_QR$, its training risk $\mathbb E_Q\hat R_n$.
- $\mathrm{KL}(Q\|P)=\mathbb E_{h\sim Q}\ln\frac{dQ}{dP}(h)\ge0$, which is $+\infty$ unless $Q$ is absolutely continuous with respect to $P$.

**Theorem** (McAllester's bound with Maurer's constants; Alquier §3.2.2,
quoted). With probability at least $1-\delta$ over $S$, simultaneously for
all $Q$,
$$
\mathbb E_QR\le\mathbb E_Q\hat R_n+\sqrt{\frac{\mathrm{KL}(Q\|P)+\ln(2\sqrt n/\delta)}{2n}} .
$$
Compare the finite-class bound $R(h)\le\hat R_n(h)+\sqrt{(\ln|\mathcal H|+\ln(1/\delta))/(2n)}$:
the class size has become a divergence of what you chose after seeing the data
from what you fixed before. The bound is for the randomized predictor, not for
the network at the centre of $Q$.

**6.1. Change of measure, and a PAC-Bayes bound in five steps.** Parts 2–3
prove Catoni's bound (Alquier, Theorem 2.1): for any $\lambda>0$ fixed before
seeing the data, with probability at least $1-\delta$, simultaneously for all
$Q$,
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
- *Continuous weights:* if $P$ has a density, then $P(\{w\})=0$ and $\mathrm{KL}(\delta_w\|P)=\infty$, so the bound is vacuous. To pay a finite price for a real-valued $w$, either store it to finite precision (a code, as above) or spread $Q$ over a neighbourhood of $w$ whose prior mass is not too small. 6.2 does the second.
"""
    ),
    markdown(
        r"""
**6.2. Gaussian posteriors: weight norm and flatness.** Take network weights
$w\in\mathbb R^p$, the prior $P=\mathcal N(w_0,\sigma_P^2I)$, with $w_0$ and
$\sigma_P$ fixed before seeing the data (for example $w_0=0$, or the random
initialization), and the posterior $Q=\mathcal N(w,\Sigma_Q)$ centred on the
trained weights $w$. Use Catoni's bound $(\star)$ from 6.1 with $\lambda$ fixed
in advance. For parts 2–4, treat $\hat R_n$ as smooth near $w$, with Hessian
$\hat H=\nabla^2\hat R_n(w)$. This holds for a bounded smooth surrogate loss;
for the 0–1 loss these parts are heuristic.

1. Show that for Gaussians in $\mathbb R^p$, $\mathrm{KL}\big(\mathcal N(\mu_Q,\Sigma_Q)\,\|\,\mathcal N(\mu_P,\Sigma_P)\big)=\tfrac12\big[\mathrm{tr}(\Sigma_P^{-1}\Sigma_Q)+(\mu_Q-\mu_P)^\top\Sigma_P^{-1}(\mu_Q-\mu_P)-p+\ln\frac{\det\Sigma_P}{\det\Sigma_Q}\big]$. Deduce that, with $s_1,\dots,s_p$ the eigenvalues of $\Sigma_Q$, $\mathrm{KL}(Q\|P)=\frac{\|w-w_0\|^2}{2\sigma_P^2}+\frac12\sum_{k=1}^p\Big(\frac{s_k}{\sigma_P^2}-1-\ln\frac{s_k}{\sigma_P^2}\Big)$. Interpret the two terms. What does the second cost for $\Sigma_Q=\sigma_Q^2I$ with $\sigma_Q\ll\sigma_P$, and how big is that for $p=10^6$ and $\sigma_P/\sigma_Q=10^3$?
2. Show that $\mathbb E_Q\hat R_n=\hat R_n(w)+\tfrac12\mathrm{tr}(\hat H\Sigma_Q)$ up to fourth-order terms in the width of $Q$, whether or not $\nabla\hat R_n(w)=0$.
3. Insert parts 1 and 2 into $(\star)$ and minimize over $\Sigma_Q$, assuming $\hat H\succeq0$ and that the quadratic approximation holds over the width of $Q$. Show that the optimum is $\Sigma_Q=(\sigma_P^{-2}I+\lambda\hat H)^{-1}$ and that the bound becomes $\mathbb E_QR\lesssim\hat R_n(w)+\frac1\lambda\Big[\frac{\|w-w_0\|^2}{2\sigma_P^2}+\frac12\ln\det\big(I+\lambda\sigma_P^2\hat H\big)+\ln\frac1\delta\Big]+\frac\lambda{8n}$. Which eigendirections of $\hat H$ cost complexity, and how much? Where does weight decay appear?
4. *The prior picks a metric.* Take the two-layer scalar linear network $f(x)=w_2w_1x$ with $\hat R_n(w)=a(w_1w_2-c^*)^2+\text{const}$, $a>0$, $c^*\ne0$ (as in 4.5.3, with the training risk in place of the population risk). Its minima form the hyperbola $w_1w_2=c^*$; at a minimum $\hat H=2a\,vv^\top$ with $v=(w_2,w_1)^\top$. The rescaling $(w_1,w_2)\to(\alpha w_1,w_2/\alpha)$ does not change the function. (Illustrative only: the squared loss needs a bounded surrogate, and the quadratic approximation needs $\sigma_P\ll|w|$.) With $w_0=0$, show that the norm term and the log-det term of part 3 depend on $\alpha$ only through $\rho(\alpha)=\alpha^2w_1^2+w_2^2/\alpha^2$, so the bound is smallest at the balanced representative and unbounded as $\alpha\to0$ or $\infty$. Is the bound then wrong, or only loose? Show that $\mathrm{KL}(Q\|P)$ is unchanged when the same smooth bijection of weight space is applied to both $P$ and $Q$. Where did the dependence on the representative come from?
5. *Random labels.* Replace the labels by labels drawn uniformly from 10 classes, independently of $x$, and use the 0–1 loss. Using the theorem in the setup, show that with probability at least $1-\delta$, every $Q$ with $\mathbb E_Q\hat R_n\le0.1$ has $\mathrm{KL}(Q\|P)\ge2n(0.8)^2-\ln(2\sqrt n/\delta)$. Evaluate this for $n=5\times10^4$ and $\delta=0.05$. Why is this consistent with SGD fitting random labels? Why does it not run into the argument of 1.1.3, that every bound depending only on the class, $n$ and $\delta$ must be at least $0.9$?

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

The bound counts the curved directions, not the parameters. This is the PAC-Bayes form of the old minimum-description-length argument that flat minima need fewer bits. Weight decay is the norm term, with $w$ held fixed here. Optimizing $w$ as well trades $\hat R_n(w)$ against $\|w-w_0\|^2/(2\lambda\sigma_P^2)$, which is weight decay towards $w_0$. Neither $\sigma_P$ nor $\lambda$ may be tuned on the data for free; both need a grid and a union bound, as in 6.1.4. (The Laplace approximation to the Bayesian evidence has the same log-determinant, with $n$ in place of $\lambda$.)

**4.** Write $\rho(\alpha)=\alpha^2w_1^2+w_2^2/\alpha^2$. The norm term is $\|w\|^2/(2\sigma_P^2)=\rho(\alpha)/(2\sigma_P^2)$. At the rescaled point $\hat H=2a\,v_\alpha v_\alpha^\top$ with $v_\alpha=(w_2/\alpha,\alpha w_1)^\top$. This has rank one with nonzero eigenvalue $2a\|v_\alpha\|^2=2a\rho(\alpha)$, so $\ln\det(I+\lambda\sigma_P^2\hat H)=\ln\big(1+2a\lambda\sigma_P^2\rho(\alpha)\big)$. Both terms increase with $\rho$. By AM–GM, $\rho\ge2|w_1w_2|=2|c^*|$, with equality iff $\alpha^2=|w_2/w_1|$, the balanced point, and $\rho\to\infty$ as $\alpha\to0$ or $\infty$.

*Wrong or loose?* Every one of these bounds is true, because the theorem holds for all $Q$ at once. They are different numbers for the same function. Since $Q$ may be chosen after seeing the data, you are free to centre it at the balanced representative, so the rescaling cannot make the best available bound worse. Dinh et al.'s construction (sharp minima, same function) shows only that the representative SGD lands on need not be the one that gives the best bound.

*Invariance.* If $\varphi$ is a smooth bijection and $P'=\varphi_\#P$, $Q'=\varphi_\#Q$, both densities pick up the same Jacobian factor, which cancels in $q'/p'$, so $\mathrm{KL}(Q'\|P')=\mathrm{KL}(Q\|P)$. The dependence on the representative came from choosing isotropic Gaussians *in the $w$ coordinates*: an isotropic Gaussian in rescaled coordinates is a different prior, which favours different representatives. Choosing $P$ is choosing a metric, and so a notion of "flat", which is the requirement 4.5.4 found for any meaningful flatness claim. Because the loss depends on $w$ only through the function $f_w$, the theorem applies equally to distributions over functions, and by the data-processing inequality their KL is no larger than the weight-space KL. Weight-space KL overcharges by paying to specify which of many equivalent $w$ you hold (Dziugaite and Roy, Appendix B).

**5.** For any fixed $h$, $\Pr[h(x)\ne y]=0.9$, because $y$ is uniform on 10 classes and independent of $x$, hence of $h(x)$. So $\mathbb E_QR=0.9$ for every $Q$. On the theorem's event, $0.9\le\mathbb E_Q\hat R_n+\sqrt{(\mathrm{KL}+\ln(2\sqrt n/\delta))/(2n)}$. If $\mathbb E_Q\hat R_n\le0.1$, then $\mathrm{KL}(Q\|P)\ge2n(0.8)^2-\ln(2\sqrt n/\delta)$. For $n=5\times10^4$ and $\delta=0.05$, $\ln(2\sqrt n/\delta)\approx9.1$, so $\mathrm{KL}\ge6.4\times10^4$ nats, about $1.3$ nats per training example.

This is consistent with fitting random labels. SGD finds $w$ with $\hat R_n(w)=0$, but every $Q$ that keeps the Gibbs training error below $0.1$ must be far from $P$: narrow around a sharp memorizing solution, or far from $w_0$, or both. The theorem predicts its own vacuity on such data. On real labels nothing forbids a small KL. Dziugaite and Roy found $Q$ with $\mathrm{KL}\approx5\text{–}9\times10^3$ nats at $n=55{,}000$, and nonvacuous bounds. Their random-label run (two classes) ended at $\mathrm{KL}\approx2\times10^5$, with a bound of $1.35$.

The argument of 1.1.3 applies to bounds of the form $\epsilon(n,\delta)$ that depend only on the class, $n$ and $\delta$. Such a bound must also hold on the random-label distribution, so it is at least $0.9$. The PAC-Bayes right-hand side also depends on $S$ through $Q$ and $\mathbb E_Q\hat R_n$. It can be large on random labels and small on real ones, with the same prior, architecture and $n$. What the bound does not explain is *why* SGD on real data lands near wide, low-KL regions. It certifies that when it happens; it does not predict it.
"""
    ),
]
