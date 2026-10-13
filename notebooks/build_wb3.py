#!/usr/bin/env python3
"""Build WB3 (Why Did LLMs Become the Dominant Paradigm?).

Shape (trail restructure, 2026-10-09): §1 is the keystone (Gwern, "The scaling
hypothesis", paired with Sutskever's "An observation on generalization") with two
short core-idea paragraphs. §2-7 are a lightly annotated reading trail, one
sentence of purpose per section and about one sentence per reading, with the
Fermi estimates and short derivations kept in "### Exercises: ..." subsections.
Kaplan et al. 2020 is WB5's keystone and appears here only as a one-line pointer.
"""
from nbkit import answer, book, markdown, reading, references, run

CELLS = [
    # ------------------------------------------------------------------ intro
    markdown(r"""
# Workbook 3 — Why Did LLMs Become the Dominant Paradigm?

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Nobody designed language models to be general. One generic objective, predicting the next token of internet text, kept rewarding more general capability and kept improving with compute; why is one of the field's central surprises.

**Route**
1. [Keystone: the scaling hypothesis](#wb3-keystone)
2. [Why generative, why unsupervised?](#wb3-section-2)
3. [Why text and not video?](#wb3-section-3)
4. [Why next-token prediction, and why the transformer?](#wb3-section-4)
5. [The opposite thesis: symmetry as inductive bias](#wb3-section-5)
6. [Why scale? Should *you* consume more tokens?](#wb3-section-6)
7. [Simulators and personae](#wb3-section-7)
"""),

    # ------------------------------------------------------------------ §1 keystone
    markdown(r"""
<a id="wb3-keystone"></a>

## 1. Keystone: the scaling hypothesis

In 2020, just after GPT-3, Gwern argued that a simple objective plus more compute was the road to general capability; the next five years largely bore him out.
""" + reading(
        "The scaling hypothesis",
        "https://gwern.net/scaling-hypothesis",
        "Gwern Branwen, 2020",
        "The case that networks get more general as they get bigger, that prediction loss rewards ever deeper understanding (the section \"Why does pretraining work?\": \"the last bits are deepest\"), and that scale, not new ideas, is the lever.",
        importance=5, length=3,
    ) + reading(
        "An observation on generalization",
        "https://www.youtube.com/watch?v=AKMuA_TVz3A",
        "Ilya Sutskever, Simons Institute, 2023 (talk)",
        "Why predicting unlabeled data should help with a task at all: the best compressor of the data and the task together must exploit every regularity they share, and next-token prediction approximates that compressor.",
        importance=4, length=3,
    ) + r"""
**One objective subsumes many tasks.** Next-token prediction needs no labels, so the data is nearly unlimited. A task is just a kind of text (a question then its answer), so a good enough model of text has to do the task. And the loss keeps rewarding generality: once spelling, grammar and topic are predicted, the remaining loss sits on tokens that need facts, reasoning and a model of the writer. Compressing everything people write forces a model to represent what they write about.

**Scale, not cleverness, became the lever.** Loss falls as a smooth power law in parameters, data and compute over many orders of magnitude ([WB5, section 'Keystone: scaling laws'](05_universality.html#wb5-keystone)), and a causal transformer computes the loss at every position in one parallel pass of matrix multiplications, which is what GPUs do best. So capability could be bought predictably with compute, and most of the gain since GPT-2 came from spending more of it. Why the power laws are so clean, and why lower loss tracks broad capability, are still open.
"""),

    # ------------------------------------------------------------------ §2
    markdown(r"""
<a id="wb3-section-2"></a>

## 2. Why generative, why unsupervised?

Labels and rewards carry few bits per sample and the input carries many, so most of what a system learns about the world has to come from predicting its own input.
""" + reading(
        'LeCun, "Predictive learning" (NIPS 2016 keynote)',
        "https://www.youtube.com/watch?v=Ount2Y4qxQo",
        "Yann LeCun, 2016",
        "The cake talk: prediction of the input is the cake, supervision the icing, reward the cherry, because that is where the bits are.",
        about="Or read the paragraph in Ruder's [Highlights of NIPS 2016](https://www.ruder.io/highlights-nips-2016/).",
        importance=4, length=3,
    ) + reading(
        "Language models are unsupervised multitask learners",
        "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "Radford, Wu, Child, Luan, Amodei, Sutskever (GPT-2), 2019",
        "Trained only on next-token prediction, GPT-2 does reading comprehension, translation and summarization zero-shot, badly but better with scale; §2 argues a task is just a kind of text.",
        importance=3, length=2,
    ) + reading(
        "Learning to generate reviews and discovering sentiment",
        "https://arxiv.org/abs/1704.01444",
        "Alec Radford, Rafal Jozefowicz, Ilya Sutskever, 2017",
        "A byte-level LSTM trained to predict Amazon reviews grows a single unit that tracks sentiment: an early case of prediction producing a concept nobody labelled.",
        importance=2, length=2,
    ) + r"""
Also: Karpathy, [The unreasonable effectiveness of recurrent neural networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (2015; the pre-transformer version of the surprise).

### Exercises: where the bits are

Count the bits a learner is asked to predict, under each kind of supervision.

**Exercise 2.1 (Fermi).**

(a) ImageNet has about $1.28\times10^6$ training images in $k=1000$ classes. Bound the number of bits needed to encode all its labels.

(b) How many raw bits are in one $224\times224$ RGB image at 8 bits per channel, and in all of ImageNet's images? What is the ratio to (a)? LeCun's slide credits RL with "a few bits for some samples", supervision with "10→10,000 bits per sample" and prediction with "millions of bits per sample". Which do your counts match?

(c) A language model is trained on $T=10^{12}$ tokens with a vocabulary of $V\approx5\times10^4$. Give the crude upper bound $T\log_2V$. Then give a better estimate: the number of bits by which the text still *surprises* a trained model whose cross-entropy is about $2$ nats per token (an illustrative value).
"""),
    answer(r"""
**(a)** A label from 1000 classes takes at most $\log_21000\approx10$ bits, so all of them take at most $1.28\times10^6\times10\approx1.3\times10^7$ bits, about 1.6 MB.

**(b)** $224\times224\times3\times8\approx1.2\times10^6$ bits per image, so about $1.5\times10^{12}$ bits for the dataset, about $10^5$ times the labels. The labels are the supervision layer's "10 bits per sample"; the images are the prediction layer's "millions of bits per sample".

**(c)** Crude bound: $\log_2(5\times10^4)\approx15.6$ bits per token, so about $1.6\times10^{13}$ bits. Residual surprise: $2/\ln2\approx2.9$ bits per token, so about $2.9\times10^{12}$ bits, with a target at every one of the $10^{12}$ positions.

These are code lengths, not measures of what a learner extracts: the residual surprise in (c) includes irreducible noise, and most of the raw bits in (b) are irrelevant to any given task. [Section 3](#wb3-section-3) is about that gap.
"""),

    # ------------------------------------------------------------------ §3
    markdown(r"""
<a id="wb3-section-3"></a>

## 3. Why text and not video?

Video has far more bits than text, but most are texture and noise, while text is data a mind has already abstracted; LeCun's JEPA is the serious attempt to get that abstraction from video instead.
""" + reading(
        "A path towards autonomous machine intelligence",
        "https://openreview.net/forum?id=BZ5a1r-kVsf",
        "Yann LeCun, 2022 (position paper)",
        "Argues that generating raw observations is the wrong objective for world models, and proposes JEPA: predict the *representation* of the missing input.",
        importance=4, length=2,
    ) + reading(
        "V-JEPA 2: self-supervised video models enable understanding, prediction and planning",
        "https://arxiv.org/abs/2506.09985",
        "Assran, Bardes, Fan, et al. (Meta), 2025",
        "JEPA at scale on a million hours of video: good motion understanding and zero-shot robot planning toward image goals, but question answering still goes through a language model.",
        importance=3, length=3,
    ) + reading(
        "Video generation models as world simulators",
        "https://openai.com/index/video-generation-models-as-world-simulators/",
        "OpenAI (Sora technical report), 2024",
        "The generative-video counterpoint: 3-D consistency and object permanence emerge from scaling a pixel generator; its own failure list includes glass shattering (compare Exercise 3.1).",
        importance=3, length=1,
    ) + r"""
### Exercises: what a bit buys

Compare text and vision first by raw volume, then by bits per event.

**Exercise 3.1 (Fermi).**

(a) Reproduce [LeCun's estimate](https://x.com/ylecun/status/1766498677751787723) (2024) of the visual data a child has received by age four: about 16,000 waking hours, through two optic nerves of about $10^6$ fibres each, carrying about 10 bytes per second per fibre.

(b) His count for a large language model's training text is $10^{13}$ tokens at 2 bytes per token. What is the ratio to (a)? How long would a person take to read that text at 270 words per minute (about 6 tokens per second, which is 12 bytes at his 2 bytes per token), 12 hours a day?

(c) Now count per event instead of per second. The sentence "The cat knocked the glass off the table and it shattered" is about 12 tokens. At an illustrative 2 nats per token for a good language model, how many bits does it cost? Compare: the same five seconds as 1080p video at 30 frames per second, raw (8 bits per channel), and as a typical stream at about 5 Mbit/s.
"""),
    answer(r"""
**(a)** $2\times10^6$ fibres $\times10$ bytes/s $=2\times10^7$ bytes/s, and $16{,}000\times3600\approx5.8\times10^7$ s, so about $1.2\times10^{15}$ bytes.

**(b)** $2\times10^{13}$ bytes, so the child has seen about 50 times more. Reading: $2\times10^{13}/12\approx1.7\times10^{12}$ s, which at 12 hours a day is about $10^5$ years. In 2025 (at NVIDIA GTC) LeCun revised the numbers to about 2 MB/s for the optic nerve and about $10^{14}$ bytes for frontier corpora, so the two come out roughly equal. The raw-byte comparison moves by orders of magnitude depending on what you call compressed, which is the point of (c).

**(c)** $12\times2/\ln2\approx35$ bits for the sentence. Raw video: $1920\times1080\times3\times8\times30\approx1.5\times10^9$ bits per second, so about $7.5\times10^9$ bits for five seconds; the stream is $5\times10^6\times5=2.5\times10^7$ bits. The compressed video is about $7\times10^5$ times longer than the sentence, the raw video about $2\times10^8$ times.

Almost none of the video's bits are about what happened; the sentence is almost nothing else. (The video does carry things the sentence omits, such as the colour of the cat.) Per bit about the variables a person cares about, text is denser by several orders of magnitude, and a pixel-predicting model spends nearly all of its loss on the rest.
"""),

    # ------------------------------------------------------------------ §4
    markdown(r"""
<a id="wb3-section-4"></a>

## 4. Why next-token prediction, and why the transformer?

Next-token prediction is a full model of the data whose loss is code length, every position is a training target, and a causal transformer computes all of them in one parallel pass, which is what the hardware rewards.

By the chain rule $q(x_{1:T})=\prod_tq(x_t\mid x_{<t})$, so this is a parametrization, not a restriction ([WB4, section 'Occam, two ways'](04_inductive_bias.html#wb4-section-2) takes up loss as code length). BERT's masked objective trains on only 15% of positions; an RNN needs $T$ sequential steps per layer where attention needs none ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762), Table 1).
""" + reading(
        "Generalized language models",
        "https://lilianweng.github.io/posts/2019-01-31-lm/",
        "Lilian Weng, 2019, updated to 2021 (blog)",
        "A tour of pretraining objectives from ELMo to GPT-3; the sections on GPT's decoder-only language model and BERT's pretraining tasks give the contrast.",
        importance=4, length=2,
    ) + reading(
        "The hardware lottery",
        "https://arxiv.org/abs/2009.06489",
        "Sara Hooker, 2020",
        "Research ideas often win because they suit the available hardware rather than because they are better; it predates the question, so apply it to transformers yourself.",
        importance=4, length=2,
    ) + reading(
        "What language model architecture and pretraining objective work best for zero-shot generalization?",
        "https://arxiv.org/abs/2204.05832",
        "Wang, Roberts, Hesslow, Le Scao, Chung, Beltagy, Launay, Raffel, 2022",
        "Controlled comparison at 5B parameters: causal next-token models generalize best zero-shot after pretraining alone, masked objectives after multitask fine-tuning; next-token prediction won the regime that turned out to matter.",
        importance=3, length=3,
    ) + reading(
        "Language modeling is compression",
        "https://arxiv.org/abs/2309.10668",
        "Grégoire Delétang, Anian Ruoss, et al. (DeepMind), 2023",
        "LLMs inside an arithmetic coder compress text, image patches and audio better than gzip, PNG and FLAC, until the model's own size is counted; §2 sets out the equivalence.",
        importance=3, length=1,
    ) + r"""
Also: Horace He, [Making deep learning go brrrr](https://horace.io/brrr_intro.html) (where GPU time goes); Jay Alammar, [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/) (if you have not met the architecture); Devlin et al., [BERT](https://arxiv.org/abs/1810.04805) (§3.1 defines the masked objective).
"""),

    # ------------------------------------------------------------------ §5
    markdown(r"""
<a id="wb3-section-5"></a>

## 5. The opposite thesis: symmetry as inductive bias

Geometric deep learning says to build the domain's symmetries into the network for sample efficiency; the transformer has little such structure, and at scale the missing bias stopped mattering.
""" + reading(
        "Geometric deep learning: the Erlangen programme of ML",
        "https://www.youtube.com/watch?v=w6Pw4MOzMuo",
        "Michael Bronstein, ICLR 2021 keynote",
        "The blueprint: choose the symmetry group, use equivariant layers, pool locally, read out invariantly; CNNs, GNNs, DeepSets and transformers are instances.",
        importance=5, length=2,
    ) + reading(
        "An image is worth 16x16 words: transformers for image recognition at scale",
        "https://arxiv.org/abs/2010.11929",
        "Dosovitskiy et al. (ViT), 2020",
        "A plain transformer on image patches trails ResNets on ImageNet alone but matches or beats them when pretrained on 14M–300M images: \"large scale training trumps inductive bias.\"",
        importance=4, length=3,
    ) + reading(
        "ConvNets match vision transformers at scale",
        "https://arxiv.org/abs/2310.16764",
        "Smith, Brock, Berrada, De (DeepMind), 2023",
        "The other half: at matched compute on JFT-4B, NFNet CNNs match ViTs with the same kind of scaling law, so neither inductive bias wins.",
        importance=3, length=2,
    ) + references(
        book("Deep Learning", "https://www.deeplearningbook.org/contents/convnets.html",
             "Goodfellow, Bengio, Courville", "ch. 9, §9.2 and §9.4 (convolution as an infinitely strong prior)"),
        book("Geometric deep learning: grids, groups, graphs, geodesics, and gauges", "https://arxiv.org/abs/2104.13478",
             "Bronstein, Bruna, Cohen, Veličković", "§3 (the blueprint is §3.5)"),
    ) + r"""
### Exercises: two symmetries

Derive the CNN from translation symmetry, then find the transformer's symmetry.

**Exercise 5.1 (convolution from symmetry).** Work with 1-D signals $x\in\mathbb R^n$, indices mod $n$, and the cyclic shift $(Sx)_i=x_{i-1}$.

(a) Let $W$ be an $n\times n$ matrix with $WS=SW$. Show that there is a vector $w$ with $(Wx)_i=\sum_jw_{i-j}x_j$ for all $x$: $W$ is a circular convolution. Show conversely that every circular convolution commutes with $S$.

(b) For a single-channel $224\times224$ image, compare the number of weights in a general linear map to an output of the same size, in a translation-equivariant one, and in a $3\times3$ convolution.

<details><summary>Hint</summary>

Set $w=We_0$ and note that $e_j=S^je_0$.

</details>
"""),
    answer(r"""
**(a)** Set $w=We_0$. Since $e_j=S^je_0$ and $W$ commutes with every power of $S$, $We_j=S^jWe_0=S^jw$, and $(S^jw)_i=w_{i-j}$. By linearity, $(Wx)_i=\sum_jx_j(We_j)_i=\sum_jw_{i-j}x_j$. Conversely, if $(Wx)_i=\sum_jw_{i-j}x_j$, then with $j'=j-1$,
$$
(WSx)_i=\sum_jw_{i-j}x_{j-1}=\sum_{j'}w_{(i-1)-j'}x_{j'}=(Wx)_{i-1}=(SWx)_i.
$$
So weight sharing is a consequence of the symmetry, not a trick: a shift-commuting linear map is fixed by $n$ numbers instead of $n^2$. Small kernels (locality) are a further, separate choice.

**(b)** With $n=224^2\approx5\times10^4$ pixels: a general linear map has $n^2\approx2.5\times10^9$ weights, a translation-equivariant one $n\approx5\times10^4$, a $3\times3$ convolution 9. (Real CNNs pad and stride, so the equivariance is only approximate.)
"""),
    markdown(r"""
**Exercise 5.2 (the transformer's symmetry).** Single-head self-attention on $T$ tokens, stacked as the rows of $X\in\mathbb R^{T\times d}$, is
$$
\mathrm{SA}(X)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,\qquad Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$
with the softmax taken along each row.

(a) Let $P$ be a $T\times T$ permutation matrix. Show that $\mathrm{SA}(PX)=P\,\mathrm{SA}(X)$.

(b) So without position information a transformer treats its input as a set. Which ingredients break the symmetry: absolute position embeddings ($X\to X+E$), a causal mask, rotary embeddings?

<details><summary>Hint</summary>

Show that $\operatorname{softmax}(PAP^\top)=P\operatorname{softmax}(A)P^\top$ for any square $A$, and use $P^\top P=I$.

</details>
"""),
    answer(r"""
**(a)** Replacing $X$ by $PX$ gives $Q'=PQ$, $K'=PK$, $V'=PV$, so the score matrix becomes $PAP^\top$ with $A=QK^\top/\sqrt{d_k}$. Row $i$ of $PAP^\top$ is a row of $A$ with its entries permuted, and a row softmax does not care about the order of entries within a row, so $\operatorname{softmax}(PAP^\top)=P\operatorname{softmax}(A)P^\top$. Then
$$
\mathrm{SA}(PX)=P\operatorname{softmax}(A)P^\top PV=P\,\mathrm{SA}(X).
$$
Position-wise MLPs and residual connections are also permutation-equivariant, so the whole stack is.

**(b)** Absolute embeddings break it: $PX+E\ne P(X+E)$ unless $PE=E$. A causal mask breaks it too, since it singles out an order, so a decoder-only transformer has some access to position even without embeddings. Rotary embeddings break permutation symmetry but make scores depend on positions only through their differences, so they keep translation symmetry in position. In the geometric picture the transformer is message passing on the *complete* graph with data-dependent edge weights. Once position is added, what remains is weight sharing across positions and content-based routing, with no locality: much less structure than a CNN's.
"""),

    # ------------------------------------------------------------------ §6
    markdown(r"""
<a id="wb3-section-6"></a>

## 6. Why scale? Should *you* consume more tokens?

Loss falls as a power law in model size and data, so a compute budget has an optimal split between the two, and serving costs push that split toward smaller models trained on more tokens.

Kaplan et al. 2020, [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361), is the keystone of [WB5, section 'Keystone: scaling laws'](05_universality.html#wb5-keystone); here it appears only as the allocation that Chinchilla corrected (Exercise 6.2).
""" + reading(
        "Training compute-optimal large language models",
        "https://arxiv.org/abs/2203.15556",
        "Hoffmann, Borgeaud, Mensch, et al. (DeepMind, \"Chinchilla\"), 2022",
        "At fixed compute, scale parameters and tokens together; Chinchilla (70B parameters, 1.4T tokens) beats the 280B Gopher at the same compute.",
        about="§3.3 and Appendix D.2 are the fit of Exercise 6.1.",
        importance=5, length=2,
    ) + reading(
        "chinchilla's wild implications",
        "https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications",
        "nostalgebraist, 2022 (LessWrong)",
        "Plugs the Chinchilla fit into how much text exists and concludes that data, not parameters, becomes the binding constraint.",
        importance=3, length=2,
    ) + r"""
Also: EleutherAI, [Transformer math 101](https://blog.eleuther.ai/transformer-math/) (derives $C\approx6ND$ and training memory).

### Exercises: compute-optimal allocation

Derive the Chinchilla allocation from a fitted loss, see why Kaplan et al. got a different answer, and price in inference.

**Exercise 6.1.** Hoffmann et al.'s third approach fits the loss of a model with $N$ parameters trained on $D$ tokens as
$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},
$$
where $E$ is the irreducible loss. Training costs about 6 FLOPs per parameter per token, so $C\approx6ND$.

(a) Minimize $L$ at fixed $C$ and show that
$$
N_{\text{opt}}=G\Big(\frac C6\Big)^{\frac{\beta}{\alpha+\beta}},\qquad
D_{\text{opt}}=G^{-1}\Big(\frac C6\Big)^{\frac{\alpha}{\alpha+\beta}},\qquad
G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac1{\alpha+\beta}}.
$$
If $\alpha>\beta$, which of $N$ and $D$ gets more of the budget?

(b) Show that at the optimum $\alpha\,AN^{-\alpha}=\beta\,BD^{-\beta}$.

(c) Show that the optimal reducible loss $L-E$ is a power law in $C$, and find its exponent.

(d) The published fit (Hoffmann et al., eq. 10 in Appendix D.2) is $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$. At Gopher's and Chinchilla's budget, $C=5.76\times10^{23}$ FLOPs, compute $N_{\text{opt}}$, $D_{\text{opt}}$ and $D_{\text{opt}}/N_{\text{opt}}$. Compare with Chinchilla's actual 70B parameters and 1.4T tokens.

(e) Under the same fit, evaluate $L$ at your $(N_{\text{opt}},D_{\text{opt}})$ and at $(7\times10^{10},\,1.4\times10^{12})$. What does the difference say about how sharply the data pin down the allocation?

<details><summary>Hints</summary>

(a) Substitute $D=C/(6N)$, set $dL/dN=0$, and collect powers of $N$. (b) Multiply the stationarity condition by $N$. (c) Substitute (a) into each reducible term. (d), (e) Work in $\log_{10}$.

</details>
"""),
    answer(r"""
**(a)** With $D=C/(6N)$,
$$
L(N)=E+AN^{-\alpha}+B\Big(\frac{6N}{C}\Big)^{\beta}.
$$
Setting the derivative to zero,
$$
-\alpha AN^{-\alpha-1}+\beta B\Big(\frac6C\Big)^{\beta}N^{\beta-1}=0
\quad\Longrightarrow\quad
N^{\alpha+\beta}=\frac{\alpha A}{\beta B}\Big(\frac C6\Big)^{\beta},
$$
which gives $N_{\text{opt}}$. Then $D_{\text{opt}}=C/(6N_{\text{opt}})=G^{-1}(C/6)^{1-\beta/(\alpha+\beta)}=G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$. ($L(N)$ is a sum of a decreasing and an increasing power of $N$, convex in $\log N$, so this stationary point is the minimum.) If $\alpha>\beta$ then $\beta/(\alpha+\beta)<\tfrac12$, so $D$ gets more of the budget: the model-size term decays faster, so it needs less help. If $\alpha=\beta$, both grow as $\sqrt C$ and $D/N$ is constant.

**(b)** Multiplying the stationarity condition by $N$ gives $\alpha AN^{-\alpha}=\beta B(6N/C)^{\beta}=\beta BD^{-\beta}$. Along the constraint $\log N+\log D=\text{const}$, these are the rates at which the two terms fall per unit of $\log N$ and of $\log D$, and at the optimum they balance.

**(c)** $AN_{\text{opt}}^{-\alpha}\propto C^{-\alpha\beta/(\alpha+\beta)}$ and $BD_{\text{opt}}^{-\beta}\propto C^{-\beta\alpha/(\alpha+\beta)}$, so $L_{\text{opt}}-E\propto C^{-\alpha\beta/(\alpha+\beta)}$. With the constants in (d) the exponent is $0.34\times0.28/0.62\approx0.15$.

**(d)** $\alpha A\approx138.2$ and $\beta B\approx115.0$, with ratio $1.202$, so $G=1.202^{1/0.62}\approx1.34$. The exponent is $\beta/(\alpha+\beta)\approx0.452$ and $C/6=9.6\times10^{22}$, so $(C/6)^{0.452}\approx10^{10.38}\approx2.4\times10^{10}$. Hence
$$
N_{\text{opt}}\approx3.2\times10^{10},\qquad
D_{\text{opt}}=\frac{9.6\times10^{22}}{3.2\times10^{10}}\approx3.0\times10^{12},
$$
about 90 tokens per parameter. Chinchilla itself used 70B parameters and 1.4T tokens, 20 tokens per parameter, an allocation that came from the paper's first two approaches. Besiroglu et al. ([Chinchilla scaling: a replication attempt](https://arxiv.org/abs/2404.10102), 2024) re-fitted the third approach to data reconstructed from the paper's plots and found the published constants fit those data poorly, while their re-fit agrees with the first two approaches.

**(e)** At the optimum, $AN^{-\alpha}\approx0.109$ and $BD^{-\beta}\approx0.132$, so $L\approx1.931$. (Check (b): $0.34\times0.109\approx0.28\times0.132\approx0.037$.) At 70B parameters and 1.4T tokens, $AN^{-\alpha}\approx0.084$ and $BD^{-\beta}\approx0.163$, so $L\approx1.937$, using about 2% more compute. A factor of more than four in tokens per parameter changes the predicted loss by about $0.006$ nats. The loss is very flat near the optimum, so small changes in the fitted constants move the optimal allocation a lot; Besiroglu et al.'s re-fit ($\alpha\approx0.35$, $\beta\approx0.37$) brings it back to about 20 tokens per parameter. Treat "20 tokens per parameter" as an order of magnitude.
"""),
    markdown(r"""
**Exercise 6.2 (why Kaplan differed).** For a loss $L=E+AN^{-\alpha}+BD^{-\beta}$ at compute $C\approx6ND$, the compute-optimal model size grows as $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$ (Exercise 6.1(a)). Kaplan et al. (2020) found $N_{\text{opt}}\propto C^{0.73}$; Hoffmann et al. (2022) found $N_{\text{opt}}\propto C^{0.5}$.

(a) Reading each result through $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$, what ratio $\beta/\alpha$ does each imply?

(b) Kaplan et al. trained every model with the same learning-rate schedule, sized for about 130B tokens, and used losses from intermediate checkpoints. At a checkpoint after $D<130$B tokens, is the recorded loss above or below what a run scheduled to end at $D$ would reach? Small compute budgets are best spent on few tokens and large ones on many. Use this to explain which way the bias pushes the fitted growth of $N_{\text{opt}}$ with $C$.
"""),
    answer(r"""
**(a)** If $\beta/(\alpha+\beta)=a$ then $\beta/\alpha=a/(1-a)$. Kaplan: $0.73/0.27\approx2.7$. Chinchilla: $\beta/\alpha\approx1$. This is only a formal reading: Kaplan et al.'s $0.73$ came from fitting loss against model size and training steps with training stopped well before convergence, not from this formula.

**(b)** Above. At an intermediate checkpoint the learning rate has not been annealed, and the final decay is where a run gains much of its last improvement. So short training looks worse than it is, most of all at small $D$, where the best allocations for small budgets lie. At small $C$ the fit is pushed toward smaller models trained on more tokens than optimal; at large $C$ it is roughly right. An optimal $N$ that is too small at small $C$ and about right at large $C$ appears to grow faster with $C$ than it really does. Hoffmann et al.: using these intermediate losses "results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size as compute budget increases."
"""),
    markdown(r"""
**Exercise 6.3 (Fermi: inference).** A model will answer $10^{12}$ queries of 1000 generated tokens each, at about $2N$ FLOPs per generated token for $N$ parameters. Compare the inference compute of a 70B model with Chinchilla's training compute ($\approx5.8\times10^{23}$ FLOPs). What does this suggest about what size of model to train?
"""),
    answer(r"""
Inference costs $10^{12}\times10^{3}\times2\times7\times10^{10}=1.4\times10^{26}$ FLOPs, about 240 times the training compute. The quantity to minimize is then $C_{\text{train}}+C_{\text{inference}}$. A smaller model trained on more tokens than the Chinchilla optimum spends more training compute to reach a given loss but costs proportionally less per generated token, and Exercise 6.1(e) shows the loss penalty for moving moderately off the optimum is small. The limit is data: overtraining needs the extra tokens to exist (see nostalgebraist's post above).
"""),

    # ------------------------------------------------------------------ §7
    markdown(r"""
<a id="wb3-section-7"></a>

## 7. Simulators and personae

Next-token prediction on human text produces a model of the text's authors rather than an agent: prompting conditions on who is writing, and post-training selects one character, the assistant.
""" + reading(
        "Simulators",
        "https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators",
        "janus, 2022 (LessWrong)",
        "A self-supervised language model is a *simulator* of its training distribution that, when sampled, runs *simulacra*, which may be agentic even though the simulator is not.",
        importance=5, length=3,
    ) + reading(
        "the void",
        "https://nostalgebraist.tumblr.com/post/785766737747574784/the-void",
        "nostalgebraist, 2025",
        "The assistant is a character with a nearly empty specification, so the base model fills it from whatever its training data say about AI assistants, fiction and alignment writing included.",
        importance=4, length=3,
    ) + reading(
        "The persona selection model: why AI assistants might behave like humans",
        "https://alignment.anthropic.com/2026/psm",
        "Sam Marks, Jack Lindsey, Chris Olah (Anthropic), 2026",
        "Post-training mostly selects one of the characters pretraining taught, so human-like behaviour is the default, and narrow fine-tuning can shift which persona the model infers it is.",
        importance=4, length=3,
    ) + reading(
        "The assistant axis: situating and stabilizing the default persona of language models",
        "https://arxiv.org/abs/2601.10387",
        "Lu, Gallagher, Michala, Fish, Lindsey (Anthropic), 2026",
        "One activation direction tracks how assistant-like the current character is; drift along it predicts slipping into other characters, and capping it resists that drift and some jailbreaks.",
        importance=3, length=3,
    ) + r"""
Also: nostalgebraist, [larger language models may disappoint you](https://www.lesswrong.com/posts/pv7Qpu8WSge8NRbpB) (2021; a prompt is evidence about what document this is); Gwern, [GPT-3 creative fiction: prompts as programming](https://gwern.net/gpt-3#prompts-as-programming) (2020; models "are not 'an' agent, because they are every agent").
"""),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb3", "03_why_llms"))
