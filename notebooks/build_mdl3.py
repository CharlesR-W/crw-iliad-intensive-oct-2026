#!/usr/bin/env python3
"""Build MDL3, "Deep Learning as Program Synthesis" (stem 04_inductive_bias).

Shape (trail restructure, reviews/2026-10-09-trail-restructure-brief.md):

1. Keystone (#wb4-keystone): Furman, "Deep learning as program synthesis";
   two short core-idea paragraphs, no exercises.
2. Occam, two ways (#wb4-section-2): prior volume vs description length / MDL;
   main: WP Kolmogorov complexity, MacKay ch. 28; supplemental: Grünwald,
   Valle-Pérez; four exercises (Occam factor, Kolmogorov basics, MDL precision,
   making the notions disagree).
3. Solomonoff and its limits (#wb4-section-3): main: WP Solomonoff induction;
   supplemental: Rathmanner-Hutter, Legg, LessWrong explainer; exercises on the mixture regret bound and Legg's diagonalization.
4. The circuit prior and sparse parity (#wb4-section-4): Hanni-Vaintrob;
   the sparse-parity exercise (counting learner vs fixed kernel).
5. Hierarchical representations (#wb4-section-5): the Random Hierarchy Model
   (Poggio et al. in "Also:"); one estimation exercise.
6. The accounts side by side (#wb4-section-6): one compact table.

Each section opens with a one-sentence purpose, then a lightly annotated reading
list, then "### Exercises: ..." whose statements carry their own notation.
The pre-restructure builder is in _pre_trail_2026-10-09/build_wb4.py.
"""
from nbkit import answer, markdown, reading, run, supplemental

CELLS = [
    # ------------------------------------------------------------------ intro
    markdown(
        r"""
# MDL3 — Deep Learning as Program Synthesis

**Mysteries of Deep Learning, Iliad Intensive 2026**

"Networks generalize because of their inductive bias" is true and explains nothing:
every learner that generalizes has an inductive bias, and naming it is a curiosity
stopper. This workbook follows one concrete proposal for what the bias is (a
preference for simple, compositional programs) and the classical theory of
simplicity it borrows from.

- [1. Keystone: deep learning as program synthesis](#wb4-keystone). Furman's hypothesis and the three paradoxes it would resolve.
- [2. Occam, two ways](#wb4-section-2). Simplicity as prior volume versus description length.
- [3. Solomonoff and its limits](#wb4-section-3). The ideal simplicity-prior learner and why no computable predictor matches it.
- [4. The circuit prior and sparse parity](#wb4-section-4). A theorem that Bayesian MLPs prefer small circuits, and what a prior does not buy.
- [5. Hierarchical representations](#wb4-section-5). Structure in the data rather than the learner.
- [6. The accounts side by side](#wb4-section-6). One table.

Exercises have folded solutions. Readings are scored for importance and length (legend
under the contents).
"""
    ),
    # ------------------------------------------------------------- keystone
    markdown(
        r"""
<a id="wb4-keystone"></a>

## 1. Keystone: deep learning as program synthesis

This is the most direct answer on offer to "why does deep learning work?": it names
the inductive bias, cites evidence for it, and says what is still unexplained.

"""
        + reading(
            "Deep learning as program synthesis",
            "https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1",
            "Zach Furman, January 2026 (LessWrong)",
            "Argues that SGD performs a tractable form of program synthesis, a search for "
            "simple compositional programs analogous to (but far cheaper than) Solomonoff "
            "induction, and names the representation and search problems a formal version "
            "must solve.",
            "About 12k words; the last third (\"The path forward\") is more speculative.",
            importance=5,
            length=3,
        )
        + r"""

**The hypothesis.** Furman's *program synthesis hypothesis*: deep learning works
because training searches for programs, meaning "solutions built by composing simple
pieces, each operating on few inputs." The architecture is a substrate, like an FPGA,
and training configures a particular program in it. The direct evidence is mechanistic
interpretability: when we can read a trained network, we tend to find algorithm-like
circuits built from reusable parts. His two worked cases are grokking on modular
addition, where test accuracy jumps from chance to near perfect as Fourier-feature
circuits form, and InceptionV1, which builds car detectors from window, body and wheel
detectors. He is explicit that this is a hypothesis: networks *can* learn such
programs, not that they always do.

**What it would explain, and what is left.** A simplicity prior over programs would turn
three paradoxes into expected outcomes. *Approximation*: the functions we care about
are compositional, and compositional functions escape the curse of dimensionality.
*Generalization*: the network that can memorize random labels finds a short program
when one exists. *Convergence*: different seeds, architectures and even brains reach
similar representations because there are few good programs to find. The open question
is then how local gradient steps on continuous parameters do a search that is
intractable by enumeration. Furman splits it into the *representation problem* (how a
discrete program sits in parameter space) and the *search problem* (how SGD finds it),
and points to singular learning theory as a route.
"""
    ),
    # ------------------------------------------------------------- section 2
    markdown(
        r"""
<a id="wb4-section-2"></a>

## 2. Occam, two ways

"Simplicity" means two different things: Bayesian evidence penalizes the prior volume
a model wastes on data that did not occur, while Kolmogorov complexity and MDL penalize
description length; they agree at leading order for regular models and part elsewhere.

For $p$ parameters and $n$ samples, both give a penalty of about $\tfrac p2\ln n$ (BIC; MDL1,
section "Bayes and MLE generalization error"). For singular models the volume penalty's $\tfrac p2$ becomes the learning
coefficient $\lambda\le\tfrac p2$ (the SLT day): the evidence measures the volume of
near-optimal parameters, not the parameter count.

"""
        + reading(
            "Kolmogorov complexity",
            "https://en.wikipedia.org/wiki/Kolmogorov_complexity",
            "Wikipedia",
            "The length of the shortest program that prints a string: invariant up to a "
            "constant under change of machine, and uncomputable.",
            importance=5,
            length=2,
        )
        + reading(
            "Model comparison and Occam's razor",
            "https://www.inference.org.uk/itprnn/book.pdf",
            "David J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, "
            "ch. 28 (2003)",
            "The Occam factor: evidence is best-fit likelihood times posterior width over "
            "prior width, so a flexible model pays for every dataset it could have fit.",
            "Chapter 28 of the free PDF; "
            "[Lecture 10](https://www.inference.org.uk/itprnn_lectures/) of the video "
            "course covers the same material.",
            importance=5,
            length=2,
        )
        + supplemental()
        + reading(
            "A tutorial introduction to the minimum description length principle",
            "https://arxiv.org/abs/math/0406077",
            "Peter Grünwald, 2004",
            "The two-part code $L(H)+L(D\\mid H)$, why it recovers BIC, and why MDL "
            "fixes a computable code class in advance.",
            "Read §1–2; §2.4.2 is the two-part code for continuous parameters.",
            importance=3,
            length=3,
        )
        + reading(
            "Deep learning generalizes because the parameter-function map is biased towards simple functions",
            "https://arxiv.org/abs/1805.08522",
            "Guillermo Valle-Pérez, Chico Q. Camargo and Ard A. Louis, ICLR 2019",
            "Bridges the two notions: the prior volume that random parameters assign to a "
            "function is exponentially biased toward low (proxy) complexity, and a "
            "PAC-Bayes bound on that prior is non-vacuous on binarized MNIST.",
            importance=2,
            length=3,
        )
        + r"""

Also: [Bayesian interpolation](https://authors.library.caltech.edu/records/r7qgh-q6g10) (MacKay 1992), the original Occam-factor paper, which ch. 28 above reworks; Wikipedia on [Minimum description length](https://en.wikipedia.org/wiki/Minimum_description_length) and the [Kraft–McMillan inequality](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality).

### Exercises: volume versus description length

These are the standard short arguments. Kolmogorov complexity is also uncomputable (the
Berry-paradox argument) and can only be approximated from above; Rathmanner and Hutter
§5.9 (in the reading list of section 3, "Solomonoff and its limits") covers this.
"""
    ),
    markdown(
        r"""
**Exercise 2.1 (The Occam factor, exactly).** A model's *evidence* is
$Z=\int p(D\mid\theta)\,\pi(\theta)\,d\theta$. Data $y_1,\dots,y_n$ are i.i.d.
$\mathcal N(\theta,\sigma^2)$ with $\sigma$ known, and the prior is
$\theta\sim\mathcal N(0,\tau^2)$. Write $\bar y$ for the sample mean and $s^2=\sigma^2/n$.

1. Show that $p(D\mid\theta)=p(D\mid\bar y)\,e^{-(\theta-\bar y)^2/2s^2}$, so the
   maximum-likelihood estimate is $\hat\theta=\bar y$.
2. Show that
   $Z=p(D\mid\hat\theta)\cdot\sqrt{\dfrac{s^2}{\tau^2+s^2}}\,\exp\!\Big(-\dfrac{\bar y^2}{2(\tau^2+s^2)}\Big)$.
   The factor after $p(D\mid\hat\theta)$ is the *Occam factor*.
3. Show that for large $n$,
   $-\ln Z=-\ln p(D\mid\hat\theta)+\tfrac12\ln n+\tfrac12\ln\dfrac{\tau^2}{\sigma^2}+\dfrac{\bar y^2}{2\tau^2}+o(1)$.
4. Fix the data and let $\tau\to\infty$. What happens to $Z$, and why does this not
   mean the model fits worse?

<details><summary>Hint</summary>

Expand $\sum_i(y_i-\theta)^2$ around $\bar y$. For step 2, write
$e^{-(\theta-\bar y)^2/2s^2}=\sqrt{2\pi s^2}\;\mathcal N(\bar y;\theta,s^2)$ and use that
convolving $\mathcal N(0,\tau^2)$ with $\mathcal N(0,s^2)$ gives $\mathcal N(0,\tau^2+s^2)$.
</details>
"""
    ),
    answer(
        r"""
**1.** $\sum_i(y_i-\theta)^2=\sum_i(y_i-\bar y)^2+n(\theta-\bar y)^2$, since the cross term contains $\sum_i(y_i-\bar y)=0$. The first part, with the normalization, is $p(D\mid\bar y)$; the second gives $e^{-(\theta-\bar y)^2/2s^2}$, maximized at $\theta=\bar y$.

**2.** $Z=p(D\mid\bar y)\int\mathcal N(\theta;0,\tau^2)\,e^{-(\theta-\bar y)^2/2s^2}d\theta=p(D\mid\bar y)\sqrt{2\pi s^2}\;\mathcal N(\bar y;0,\tau^2+s^2)$, and $\sqrt{2\pi s^2}\,(2\pi(\tau^2+s^2))^{-1/2}=\sqrt{s^2/(\tau^2+s^2)}$.

**3.** $-\ln Z=-\ln p(D\mid\hat\theta)+\tfrac12\ln(1+\tau^2/s^2)+\bar y^2/2(\tau^2+s^2)$. With $\tau^2/s^2=n\tau^2/\sigma^2$ the middle term is $\tfrac12\ln n+\tfrac12\ln(\tau^2/\sigma^2)+O(1/n)$, and the last tends to $\bar y^2/2\tau^2$. The Occam factor is $\approx\Delta_{\text{post}}/\Delta_\pi$ with posterior width $\Delta_{\text{post}}=\sqrt{2\pi}\,\sigma/\sqrt n$ and effective prior width $\Delta_\pi=1/\pi(\hat\theta)$. The $\tfrac12\ln n$ is BIC's $\tfrac p2\ln n$ with $p=1$: it comes from the posterior shrinking like $1/\sqrt n$.

**4.** The Occam factor behaves like $\sigma/(\tau\sqrt n)\to0$, so $Z\to0$ while $p(D\mid\hat\theta)$ is unchanged. The fit is as good as before; the model is penalized because a vague prior put almost all its mass on values the data ruled out. Evidence is sensitive to prior widths in a way that likelihood fits are not.
"""
    ),
    markdown(
        r"""
**Exercise 2.2 (Kolmogorov complexity: three facts).** Fix a universal prefix machine
$U$: its valid programs form a prefix-free set (no program is a proper prefix of
another). The Kolmogorov complexity of a string $x$ is $K_U(x)=\min\{|p|:U(p)=x\}$.

**(a) Kraft.** To a binary string $p=p_1\dots p_k$ assign the interval
$I_p=[0.p_1\dots p_k,\ 0.p_1\dots p_k+2^{-k})\subset[0,1)$ (a binary fraction). Show
that $I_p\cap I_q=\varnothing$ when neither of $p,q$ is a prefix of the other. Conclude
$\sum_{p\in P}2^{-|p|}\le1$ for prefix-free $P$, and hence $\sum_x2^{-K_U(x)}\le1$: a
short description and a high prior probability are the same thing in two units.

**(b) Invariance.** Universality means that for every prefix machine $V$ there is a
string $s_V$ with $U(s_Vp)=V(p)$ for all $p$. Prove $K_U(x)\le K_V(x)+|s_V|$. Then,
for a fixed string $x^\star$, build a universal prefix machine $V$ with
$K_V(x^\star)=1$. What does this say about "the Kolmogorov complexity of this trained
network"?

**(c) Counting.** Show that fewer than a $2^{-c}$ fraction of $n$-bit strings have
$K(x)<n-c$. Why does this make a preference for simplicity informative?

<details><summary>Hints</summary>

(a) If neither string is a prefix of the other, they first differ at some position $j$;
see which half of a dyadic interval of length $2^{-(j-1)}$ each lies in.
(b) Let $V$ reserve one very short program for $x^\star$ and otherwise defer to $U$.
(c) Count binary strings of length less than $n-c$.
</details>
"""
    ),
    answer(
        r"""
**(a)** Let $p,q$ first differ at position $j$, with $p_j=0$, $q_j=1$, and common prefix $a=0.p_1\dots p_{j-1}$. Then $I_p\subset[a,a+2^{-j})$ and $I_q\subset[a+2^{-j},a+2^{-j+1})$, so they are disjoint. The intervals of a prefix-free set are disjoint subintervals of $[0,1)$ with lengths $2^{-|p|}$, so their sum is at most 1. Distinct strings have distinct shortest programs, all in the prefix-free domain of $U$, so $\sum_x2^{-K_U(x)}\le1$. (Conversely, any lengths satisfying Kraft's inequality are the lengths of some prefix code, which is why $L=-\log_2P$ converts freely between code lengths and priors.)

**(b)** If $p$ is a shortest $V$-program for $x$, then $s_Vp$ is a $U$-program for $x$, so $K_U(x)\le K_V(x)+|s_V|$; swapping roles gives $|K_U-K_V|\le c$. For the construction, set $V(\texttt{0})=x^\star$ and $V(\texttt{1}p)=U(p)$. Its domain is prefix-free, it is universal ($K_V(y)\le K_U(y)+1$), and $K_V(x^\star)=1$ however large $K_U(x^\star)$ is. So the invariance constant can exceed any single object's complexity: for one finite object, such as a trained network, $K$ has no machine-independent value.

**(c)** There are $2^{n-c}-1$ binary strings shorter than $n-c$, each the program for at most one output, so fewer than $2^{n-c}$ strings have $K<n-c$: a fraction below $2^{-c}$ of the $2^n$ strings. Fewer than one $n$-bit string in a thousand compresses by 10 bits. Simple strings are rare, so a learner that bets on them is making a strong, falsifiable bet.
"""
    ),
    markdown(
        r"""
**Exercise 2.3 (Why MDL states parameters to precision $1/\sqrt n$).** The MDL
two-part code scores a hypothesis by $L(H)+L(D\mid H)$ bits, with
$L(D\mid H)=-\log_2p(D\mid H)$. Take one parameter $\theta$ in an interval of length
$\Delta_\pi$. Near the maximum-likelihood estimate,
$-\ln p(D\mid\theta)\approx-\ln p(D\mid\hat\theta)+\tfrac{nJ}{2}(\theta-\hat\theta)^2$,
with $J$ the Fisher information per sample, known to sender and receiver along with
$n$. The code rounds $\hat\theta$ to the nearest point $\theta_q$ of a grid with
spacing $\delta$, sends its index ($\log_2(\Delta_\pi/\delta)$ bits), then sends the data
using $p(\cdot\mid\theta_q)$.

1. Taking $\theta_q-\hat\theta$ uniform on $[-\delta/2,\delta/2]$, write the expected
   total length $L(\delta)$ in bits and minimize it.
2. Show $L(\delta^\ast)=-\log_2p(D\mid\hat\theta)+\tfrac12\log_2n+O(1)$. What does the
   $O(1)$ depend on?
3. Why does the optimal $\delta$ shrink like $1/\sqrt n$?

<details><summary>Hint</summary>

$\mathbb E[(\theta_q-\hat\theta)^2]=\delta^2/12$. Divide nats by $\ln2$ to get bits.
</details>
"""
    ),
    answer(
        r"""
**1.** $L(\delta)=\log_2(\Delta_\pi/\delta)-\log_2p(D\mid\hat\theta)+\dfrac{nJ\delta^2}{24\ln2}$. Setting $dL/d\delta=0$ gives $\delta^\ast=\sqrt{12/(nJ)}$.

**2.** At $\delta^\ast$ the index costs $\log_2\Delta_\pi+\tfrac12\log_2n+\tfrac12\log_2(J/12)$ bits and the fit penalty is $1/(2\ln2)\approx0.72$ bits. The $O(1)$ is $\log_2\Delta_\pi+\tfrac12\log_2(J/12)+1/(2\ln2)$: it depends on the parameter's range and its Fisher information, and so on the coordinates chosen for $\theta$. With $p$ parameters (Fisher matrix diagonal) the total is $\tfrac p2\log_2n$, which is BIC in bits.

**3.** $\delta^\ast\approx3.5/\sqrt{nJ}$ is a few posterior standard deviations. Stating $\theta$ more precisely than the data can resolve costs bits and buys no fit; stating it less precisely costs fit. The $\tfrac12\log_2n$ is the code-length counterpart of the Occam factor $\Delta_{\text{post}}/\Delta_\pi$ in Exercise 2.1.
"""
    ),
    markdown(
        r"""
**Exercise 2.4 (Make them disagree).** Call a model *volume-simple* if its evidence
$Z=\int p(D\mid\theta)\pi(\theta)d\theta$ is not much below its best-fit likelihood,
i.e. its Occam factor (roughly posterior width over prior width) is not tiny. Call a
hypothesis *description-simple* if it has a short program. Give:

1. a model that is short to describe but heavily Occam-penalized;
2. a hypothesis that is short to describe but that Bayesian comparison over a
   reasonable model family will not favour;
3. a model that a code which states every parameter separately (Exercise 2.3) calls
   expensive but whose evidence is barely penalized.

<details><summary>Hint</summary>

(1) One parameter, a strange prior. (2) The digits of $\pi$. (3) A singular model: many
parameters, small learning coefficient $\lambda$.
</details>
"""
    ),
    answer(
        r"""
- **1.** $y=\theta x+\varepsilon$ with $\theta\sim\mathrm{Unif}[-10^{100},10^{100}]$. One line to state; but with $\Delta_{\text{post}}\approx10^{-2}$ the Occam factor is about $10^{-102}$, roughly 340 bits. Volume cares about the prior's spread, not the model's description.
- **2.** The first $10^6$ binary digits of $\pi$. $K$ is at most a few thousand bits (a short program plus about 20 bits for the length). But i.i.d. and order-$k$ Markov models all see a fair coin, and the best evidence goes to the fair coin at about $10^6$ bits. Bayesian comparison only ranks the models you put in; Kolmogorov complexity ranks against every program.
- **3.** A singular model with $p$ parameters and learning coefficient $\lambda\ll p/2$ pays about $\lambda\ln n$ nats in volume, while a code that quantizes every coordinate pays about $\tfrac p2\ln n$. Degenerate directions cost the evidence nothing. A cleverer code that states only the non-degenerate directions closes some of the gap; the evidence does it automatically.

So when a paper says "networks are biased toward simple functions," ask which notion it means.
"""
    ),
    # ------------------------------------------------------------- section 3
    markdown(
        r"""
<a id="wb4-section-3"></a>

## 3. Solomonoff and its limits

Solomonoff induction, the ideal that Furman's hypothesis borrows from, is a Bayesian
mixture over all programs weighted by $2^{-\text{length}}$; its regret against any computable
source is at most that source's description length, but it is uncomputable, and no simple
predictor approximates it.

"""
        + reading(
            "Solomonoff's theory of inductive inference",
            "https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference",
            "Wikipedia",
            r"The universal prior $2^{-\text{length}}$ over programs, the Bayes mixture it "
            r"defines, its convergence on computable sources, and why it is uncomputable.",
            importance=5,
            length=2,
        )
        + supplemental()
        + reading(
            "A philosophical treatise of universal induction",
            "https://arxiv.org/abs/1105.5721",
            "Samuel Rathmanner and Marcus Hutter, 2011 (*Entropy* 13(6))",
            "An accessible exposition of Solomonoff induction: Bayes mixtures and their "
            "bounds, Kolmogorov complexity, the universal prior, computable approximations "
            "and the philosophical objections.",
            "About 60 pages; for this workbook §4.7–4.9, §5.9, §6.5–8.1, §9 (MDL and "
            "approximations) and §10.2 (the reference-machine problem).",
            importance=3,
            length=3,
        )
        + reading(
            "Is there an elegant universal theory of prediction?",
            "https://arxiv.org/abs/cs/0606070",
            "Shane Legg, 2006",
            "Every computable predictor is defeated by a sequence of about its own "
            "complexity, so predictors for all sequences up to complexity $n$ must "
            "themselves be about that complex; Exercise 3.2 follows it.",
            importance=3,
            length=2,
        )
        + reading(
            "An intuitive explanation of Solomonoff induction",
            "https://www.lesswrong.com/posts/Kyc5dFDzBg4WccrbK/an-intuitive-explanation-of-solomonoff-induction",
            "Alex Altair, LessWrong, 2012",
            "A long informal walkthrough that builds the universal prior up from Occam, "
            "Bayes and Turing machines with worked examples.",
            importance=2,
            length=3,
        )
        + r"""

### Exercises: what Solomonoff guarantees and what it cannot
"""
    ),
    markdown(
        r"""
**Exercise 3.1 (The regret bound).** Take a countable class of semimeasures $\nu$ on
binary sequences (probabilities $\nu(x)$ of prefixes $x$, allowed to sum to less than 1)
with weights $w_\nu>0$, $\sum_\nu w_\nu\le1$, and form the mixture
$\xi=\sum_\nu w_\nu\nu$. It predicts with $\xi(a\mid x)=\xi(xa)/\xi(x)$. Suppose the
class contains a measure $\mu$ that generates the data. Solomonoff's mixture is the case
where the class is all lower-semicomputable semimeasures and $w_\nu=2^{-K(\nu)}$.

**(a)** Show that $\xi(x)\ge w_\mu\,\mu(x)$ for every $x$.

**(b)** Show that $\prod_{t\le n}\xi(x_t\mid x_{<t})=\xi(x_{1:n})/\xi(\epsilon)$ (with
$\epsilon$ the empty string), and conclude that the cumulative log-loss regret of $\xi$
against $\mu$ is at most $\ln(1/w_\mu)$ on every sequence and every $n$. Specialize to
$w_\mu=2^{-K(\mu)}$.

**(c)** Evaluate the bound for an environment with a 1000-bit description. How does it
compare with the cost of simply being told $\mu$?

<details><summary>Hint</summary>

Write each conditional as a ratio and let the product telescope; $\xi(\epsilon)\le1$ and
$\mu(\epsilon)=1$.
</details>
"""
    ),
    answer(
        r"""
**(a)** $\xi(x)$ is a sum of non-negative terms, one of which is $w_\mu\mu(x)$.

**(b)** The product telescopes to $\xi(x_{1:n})/\xi(\epsilon)$, and likewise for $\mu$ with $\mu(\epsilon)=1$. So the regret is

$$
R_n=\sum_{t\le n}\ln\frac{\mu(x_t\mid x_{<t})}{\xi(x_t\mid x_{<t})}=\ln\frac{\mu(x_{1:n})}{\xi(x_{1:n})}+\ln\xi(\epsilon)\le\ln\frac1{w_\mu},
$$

using $\xi(\epsilon)\le1$ and (a). With $w_\mu=2^{-K(\mu)}$ the bound is $K(\mu)\ln2$ nats, on every sequence: a worst-case guarantee, not an average. Taking expectations under $\mu$, the per-step KL divergences between $\mu$'s and $\xi$'s predictions sum to at most $K(\mu)\ln2$ over all time, so the mixture's predictions converge to the truth (Rathmanner and Hutter §8.1).

**(c)** $1000\ln2\approx693$ nats, exactly 1000 bits of total excess code length however long the sequence. The mixture pays no more, in total, than it would cost to be told the shortest description of $\mu$ up front. This is the MDL reading of Solomonoff induction. Note that the bound is generic to Bayes mixtures; Solomonoff's contribution is the class (every computable source) and the weights (which make the price the truth's description length).
"""
    ),
    markdown(
        r"""
**Exercise 3.2 (Diagonalize a predictor).** A *predictor* is a computable map $p$ from
finite binary strings to $\{0,1\}$. It *learns* an infinite sequence
$\omega=\omega_1\omega_2\dots$ if $p(\omega_{1:n})=\omega_{n+1}$ for all large enough
$n$. Write $K(\omega)$ for the length of a shortest program that outputs $\omega$ bit by
bit, and $K(p)$ for that of a shortest program computing $p$.

**(a)** Given $p$, construct a computable $\omega$ with $p(\omega_{1:n})\ne\omega_{n+1}$
for every $n$, and show $K(\omega)\le K(p)+c$ for a constant $c$ independent of $p$.

**(b)** Deduce that a predictor that learns every computable sequence with $K\le n$ has
$K(p)\ge n-c$.

**(c)** Apply (a) to a trained language model with greedy decoding. What sequence
defeats it, and how complex is it? What is the analogue under log-loss?

**(d)** There is a single simple predictor that learns sequences of arbitrarily high
complexity. Find one, and reconcile it with (b).

<details><summary>Hint</summary>

(a) Do the opposite of whatever $p$ predicts. (d) Sequences that are complicated only at
the start.
</details>
"""
    ),
    answer(
        r"""
**(a)** Set $\omega_1=1-p(\epsilon)$ and $\omega_{n+1}=1-p(\omega_{1:n})$. A program for $\omega$ is a shortest program for $p$ plus a fixed wrapper that runs it and flips its outputs, so $K(\omega)\le K(p)+c$, and $p$ is wrong at every step (Legg's Lemma 3.3).

**(b)** By (a) some $\omega'$ with $K(\omega')\le K(p)+c$ is not learned by $p$, so $K(\omega')>n$, hence $K(p)>n-c$ (Legg's Theorem 4.3).

**(c)** At each step emit a token other than the model's argmax. The model is always wrong, and the sequence is computable from the weights plus a constant-size wrapper, so its complexity is at most the model's plus a constant. Under log-loss, emit the token the model rates least likely: its probability is at most $1/V$ for vocabulary size $V$, so the model pays at least $\ln V$ nats per step. A claim that a fixed model is a universal predictor must therefore be distributional ("on the text that actually occurs") or resource-bounded, not worst-case.

**(d)** "Always predict 0" learns every sequence $x000\ldots$, and $K(x000\ldots)$ is as large as you like for incompressible $x$. No conflict: a simple predictor can learn *some* complex sequences, but no simple predictor learns *all* simple ones (Legg's Lemma 3.2).
"""
    ),
    # ------------------------------------------------------------- section 4
    markdown(
        r"""
<a id="wb4-section-4"></a>

## 4. The circuit prior and sparse parity

A theorem that Bayesian learning with wide fixed-depth MLPs has a small-circuit prior,
and a toy problem (sparse parity) that separates what such a prior buys from what a fixed
kernel can do, and from what gradient search actually achieves.

"""
        + reading(
            "A circuit prior in NN-bayes",
            "https://www.lesswrong.com/posts/SqDHeuycNkERurtSc/a-circuit-prior-in-nn-bayes",
            "Kaarel Hänni and Dmitry Vaintrob, August 2026 (LessWrong post linking "
            "[talk slides](https://kaarelh.github.io/ai/nnbayes-generalization.pdf), Iliad 2026)",
            r"Under a strong-prior Gaussian scaling, the Bayesian posterior of an "
            r"overparametrized fixed-depth MLP learns a Boolean target of fixed-depth "
            r"circuit complexity $C$ from $\mathrm{poly}(d,C)$ samples at any width "
            r"$N\ge C$; slide 34 contrasts NNGP's $\sim d^k$ for $k$-sparse parity.",
            r"Slides 10–12 are a warm-up (Solomonoff function induction), 15–17 the "
            r"theorem; a full write-up is forthcoming.",
            importance=5,
            length=2,
        )
        + supplemental()
        + reading(
            "Hidden progress in deep learning: SGD learns parities near the computational limit",
            "https://arxiv.org/abs/2207.08799",
            "Boaz Barak, Benjamin L. Edelman, Surbhi Goel, Sham Kakade, Eran Malach and Cyril Zhang, NeurIPS 2022",
            r"The empirical row of the table: SGD on small MLPs does learn $k$-sparse "
            r"parities, after a number of steps polynomial in $d$ with a $k$-dependent "
            r"exponent (roughly $d^{k-1}$ in the regime of their analysis, far below the "
            r"$\sim d^k$ kernel scale and far above $k\log_2d$); progress is invisible "
            r"to the loss until a sharp transition.",
            r"Sections 1–3 for the phenomenon and the Fourier-gap mechanism; the rest is proofs.",
            importance=3,
            length=3,
        )
        + r"""

Also: [Width-robust learnability in mean-field Bayesian neural networks](https://arxiv.org/abs/2607.05735) (Vaintrob and Hänni 2026), the mean-field counterpart; Wikipedia on [Parity learning](https://en.wikipedia.org/wiki/Parity_learning), the problem and its noisy variant.

### Exercises: sparse parity
"""
    ),
    markdown(
        r"""
**Exercise 4.1 (Counting learner versus kernel learner).** Inputs $x$ are uniform on
$\{\pm1\}^d$. The target is the parity $\chi_S(x)=\prod_{i\in S}x_i$ for an unknown
$S\subseteq\{1,\dots,d\}$ with $|S|=k$; labels are noiseless. Write $N=\binom dk$ for
the number of candidate targets.

**(a) Short-description prior.** The learner outputs any $k$-sparse parity consistent
with the $n$ training examples. Show that two distinct parities disagree on exactly half
of all inputs. Deduce that with probability at least $1-\delta$ the only consistent
parity is the target once $n\ge\log_2N+\log_2(1/\delta)$, and that $\log_2N\le k\log_2d$.

**(b) Fixed kernel.** Kernel ridge regression, minimum-norm kernel interpolation and the
Gaussian-process posterior mean all output $\hat f=\sum_{i\le n}\alpha_iK(x_i,\cdot)$,
which lies in $V=\mathrm{span}\{K(x_i,\cdot)\}$; $V$ depends on the inputs but not the
labels. In $L^2$ of the uniform distribution, show
$\sum_{|S|=k}\lVert P_V\chi_S\rVert^2\le n$ ($P_V$ the orthogonal projection onto $V$),
and deduce $\mathbb E_S\lVert\hat f-\chi_S\rVert^2\ge1-n/N$ for a uniformly random target.

**(c)** Put in $d=100$, $k=5$. What does each result say, and not say, about networks
trained by SGD?

**(d) Optional, empirical.** Train a small MLP by SGD on $k=3$ for a few values of $d$,
record the number of samples at which test error first drops well below $1/2$, and fit a
slope on log–log axes. Theory only brackets the answer, between $\log_2d$-ish (a) and
$d^k$ (b); report what you get.

<details><summary>Hints</summary>

(a) $\chi_S\chi_T=\chi_{S\triangle T}$, and a non-constant parity has mean zero; then a
union bound. (b) The $\chi_S$ are orthonormal. Expand $\lVert P_V\chi_S\rVert^2$ in an
orthonormal basis of $V$ (at most $n$ vectors), swap sums, use Bessel's inequality, and
note $\lVert\hat f-\chi_S\rVert^2\ge1-\lVert P_V\chi_S\rVert^2$.
</details>
"""
    ),
    answer(
        r"""
**(a)** For $S\ne T$, $\mathbb E_x[\chi_S\chi_T]=\mathbb E_x[\chi_{S\triangle T}]=0$, so they agree on exactly half the inputs, and a fixed wrong parity survives $n$ i.i.d. examples with probability $2^{-n}$. A union bound over the $N-1$ wrong parities gives failure probability at most $N2^{-n}\le\delta$ once $n\ge\log_2N+\log_2(1/\delta)$; and $N\le d^k$. The sample cost is the number of bits needed to name $S$: the Occam bound in miniature, and the shape of the slides' Solomonoff warm-up.

**(b)** With $e_1,\dots,e_r$ ($r\le n$) an orthonormal basis of $V$,

$$
\sum_{|S|=k}\lVert P_V\chi_S\rVert^2=\sum_{j=1}^r\sum_{|S|=k}\langle e_j,\chi_S\rangle^2\le\sum_{j=1}^r\lVert e_j\rVert^2=r\le n
$$

by Bessel. Since $\hat f\in V$, $\lVert\hat f-\chi_S\rVert^2\ge1-\lVert P_V\chi_S\rVert^2$; averaging over $S$ gives $1-n/N$. The predictor $\hat f=0$ already has error 1, so any fixed kernel needs $n$ of order $\binom dk\approx d^k/k!$ to do much better. (A posterior *sample* is the mean plus label-independent noise, which only adds error.)

**(c)** $N=\binom{100}5\approx7.5\times10^7$ and $\log_2N\approx26.2$: about 33 samples for $\delta=0.01$ versus tens of millions for any fixed kernel. That is the gap between a description-length (or circuit) prior and the NNGP/NTK kernel prior. But both numbers are theory, and neither is SGD: (a) is the information-theoretic count for a learner that enumerates $N$ candidates, (b) the scale for a learner that never moves its features. A network trained by SGD sits empirically in between. Barak et al. (2022) find that online SGD on a small MLP (one fresh sample per step) learns sparse parities after a number of steps polynomial in $d$ with a $k$-dependent exponent, roughly $d^{k-1}$ in the regime of their analysis; the exponent depends on width, learning rate and $k$, and is a measured quantity rather than a theorem. Statistical-query lower bounds say any SQ algorithm needs $d^{\Omega(k)}$ queries, so the gap between (a) and SGD is about *search*, not the prior: Furman's search problem.

**(d)** Expect a clean power law with an exponent well below $k$ and well above $0$; small runs at $k=3$ typically land nearer $d^{1}$–$d^{2}$ than $d^3$, and the number moves with width, learning rate and the threshold you choose. The point is the bracket, not the slope: SGD is neither the counting learner nor a fixed kernel.
"""
    ),
    # ------------------------------------------------------------- section 5
    markdown(
        r"""
<a id="wb4-section-5"></a>

## 5. Hierarchical representations

A different answer puts the structure in the data: compositional targets are cheap for
deep networks to approximate (Furman's paradox of approximation), and in a solvable model
the sample cost is set by when low-level correlations become detectable, not by a
simplicity prior.

"""
        + reading(
            "How deep neural networks learn compositional data: the Random Hierarchy Model",
            "https://arxiv.org/abs/2307.02129",
            "Francesco Cagnetta, Leonardo Petrini, Umberto M. Tomasini, Alessandro Favero and Matthieu Wyart, 2024 (*Phys. Rev. X* 14, 031001)",
            r"On synthetic hierarchical grammars, deep networks need about $n_cm^L$ "
            r"samples (polynomial in input dimension), shallow ones exponentially many; "
            r"removing low-level feature–label correlations makes it exponentially hard "
            r"for deep networks too.",
            "§I defines the model, §II the scaling, §III the correlation argument.",
            importance=5,
            length=3,
        )
        + r"""

Also: [Why and when can deep, but not shallow, networks avoid the curse of dimensionality: a review](https://arxiv.org/abs/1611.00740) (Poggio et al. 2017): a deep network matched to a binary tree of 2-variable functions approximates it with about $(d-1)\varepsilon^{-2/r}$ units where a shallow one needs $\varepsilon^{-d/r}$ (Theorems 1–2); an existence result about approximation, not learning.

### Exercises: where does simplicity enter?
"""
    ),
    markdown(
        r"""
**Exercise 5.1 (A length-prior learner on the Random Hierarchy Model).** In the Random
Hierarchy Model each level has $v$ symbols; each symbol at level $\ell$ is rewritten as
a patch of $s$ symbols at level $\ell-1$, with $m$ random but fixed alternative rewrites
("synonyms"), $m\le v^{s-1}$. A class label (one of $n_c$) is rewritten $L$ times, giving
an input of length $d=s^L$. Deep CNNs trained by gradient descent need about
$P^\ast\simeq n_cm^L$ samples.

Estimate how many samples an idealized learner with a program-length prior would need,
and compare with $P^\ast$ as $L$ grows. What does the comparison suggest about what sets
the observed $P^\ast$?

<details><summary>Hint</summary>

Count the bits needed to write down all the rewrite rules. A Bayesian learner over
programs needs a number of samples linear in the target's description length, and each
label carries at most $\log_2n_c$ bits.
</details>
"""
    ),
    answer(
        r"""
- **Description length.** $L$ levels, about $v$ symbols each, $m$ rewrites of $s$ symbols at $\log_2v$ bits per symbol: about $Lvms\log_2v$ bits.
- **Samples.** At most $\log_2n_c$ bits per label, so roughly $Lvms\log_2v/\log_2n_c$ samples: *linear* in $L$, whereas $P^\ast\simeq n_cm^L$ is *exponential* in $L$ (though polynomial in $d=s^L$).
- **Example.** $v=n_c=8$, $m=4$, $s=2$: at $L=3$, about 192 versus 512; at $L=6$, about 384 versus 32,768.
- **Reading.** The idealized learner is far more sample-efficient but intractable. The network's actual cost tracks when correlations become detectable, so $P^\ast$ reflects the search mechanism and the data's statistics, not a program-length prior. The uncorrelated variant sharpens this: the rules are just as short, so the idealized learner is unaffected, but deep networks become exponential.
- **Caveat.** A back-of-envelope estimate; unlabelled inputs also carry information about the rules, so the label-only count overstates what the idealized learner needs.
"""
    ),
    # ------------------------------------------------------------- section 6
    markdown(
        r"""
<a id="wb4-section-6"></a>

## 6. The accounts side by side

One table comparing the priors over functions met above on the two test problems.

| prior over functions | $k$-sparse parity on $\{\pm1\}^d$ | correlated vs uncorrelated hierarchy |
|---|---|---|
| program length, $2^{-\ell(f)}$ (sections 2–3, keystone) | $\approx k\log_2d$ samples, information-theoretic (Ex. 4.1a) | equally short, so equally easy |
| circuit size at fixed depth (Hänni–Vaintrob) | $\mathrm{poly}(d,C)$ samples, any width $N\ge C$ | not addressed |
| kernel norm (NNGP, lazy/NTK) | $\gtrsim\binom dk$ samples for any fixed kernel (Ex. 4.1b) | not addressed |
| *observed:* MLPs trained by SGD | samples between $k\log_2d$ and $d^k$: polynomial in $d$ with a $k$-dependent exponent (Barak et al. 2022, Ex. 4.1d); SQ lower bound $d^{\Omega(k)}$ queries | polynomial vs exponential in $d$ (Ex. 5.1) |

Singular learning theory is not a fourth row: it describes posterior concentration in
parameter space (section 2's volume, done correctly for singular models); see the SLT day.
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb4", "mdl3_program_synthesis"))
