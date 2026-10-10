#!/usr/bin/env python3
"""Build WB3 (Why LLMs?). Reading-centred, with Fermi estimates and a few short derivations."""
from nbkit import answer, book, markdown, reading, references, run

CELLS = [
    # ------------------------------------------------------------------ intro
    markdown(r"""
# Workbook 3 — Why LLMs?

**{{VARIANT}} · Mysteries of Deep Learning, Iliad Intensive 2026**

Deep learning reached large language models through a series of choices that each looked contingent at the time: learn without labels, learn from text, predict the next token with a transformer, and make it big. This workbook is mostly readings, one short list per choice, with a sentence or two on each so you can decide what to open. A few Fermi estimates and short derivations sit where the numbers or the algebra help. Worked answers are folded under "Solution". Sections are independent; nobody is expected to do all of it.

**Route**
1. [Why generative, why unsupervised?](#wb3-section-1) LeCun's cake, Sutskever on compression, and early demonstrations that predicting your input teaches you about the world. A Fermi estimate of where the bits are.
2. [Why text and not video?](#wb3-section-2) Text is data that a mind has already abstracted and compressed. A Fermi estimate of what a bit of each buys; JEPA as the attempt to get the same abstraction from video.
3. [Why next-token prediction, and why the transformer?](#wb3-section-3) Next-token loss is code length; every position is a training target; a causal transformer computes all of them in one parallel pass. GPT vs BERT, and the hardware lottery.
4. [A hundred flowers: symmetry as inductive bias](#wb3-section-4) The geometric-deep-learning view of CNNs and their relatives, with two short symmetry derivations. Read it as the opposite thesis to §3.
5. [Why scale? Should *you* consume more tokens?](#wb3-section-5) Kaplan and Chinchilla, with three exercises on compute-optimal allocation and inference cost.
6. [Simulators and personae](#wb3-section-6) What a base model is, and what post-training selects from it.
7. [What carries forward](#wb3-closing)
"""),

    # ------------------------------------------------------------------ §1
    markdown(r"""
<a id="wb3-section-1"></a>

## 1. Why generative, why unsupervised?

Labels and rewards carry few bits; the input itself carries many. If most of what a system must learn is the structure of the world, it has to come from predicting its own input. The readings make this case from three angles: LeCun's bit-counting, Sutskever's compression argument, and two early OpenAI results in which a model trained only to predict text turned out to have learned things nobody asked for.
""" + reading(
        'LeCun, "Predictive learning" (NIPS 2016 keynote)',
        "https://www.youtube.com/watch?v=Ount2Y4qxQo",
        "Yann LeCun, 2016",
        "The cake talk. Most of what an intelligent system learns must come from predicting its own input, because that is where most of the bits are; supervision and reward are the icing and the cherry.",
        about="If you would rather read, Sebastian Ruder's [Highlights of NIPS 2016](https://www.ruder.io/highlights-nips-2016/) describes the cake and the reaction to it in a paragraph.",
        importance=4, length=3,
    ) + reading(
        "An observation on generalization",
        "https://www.youtube.com/watch?v=AKMuA_TVz3A",
        "Ilya Sutskever, Simons Institute, 2023 (talk)",
        "Why should unsupervised learning work at all, when the objective you optimize (predict the input) is not the task you care about? Sutskever's answer is compression: a good compressor of the unlabeled data and the task data together must exploit every regularity they share. He takes Kolmogorov complexity as the ideal compressor and next-token prediction as a practical approximation to it.",
        importance=5, length=3,
    ) + reading(
        "The scaling hypothesis",
        "https://gwern.net/scaling-hypothesis#why-does-pretraining-work",
        "Gwern Branwen, 2020",
        "The section \"Why does pretraining work?\" is the classic statement of why prediction teaches so much: once a model has predicted the easy statistical regularities, what is left of the loss needs reasoning, common sense and theory of mind. \"The last bits are deepest.\"",
        about="The rest of the essay argues, in 2020, that GPT-3 showed scale itself to be the path; its appendix \"It from byte\" belongs with §6.",
        importance=4, length=1,
    ) + reading(
        "Learning to generate reviews and discovering sentiment",
        "https://arxiv.org/abs/1704.01444",
        "Alec Radford, Rafal Jozefowicz, Ilya Sutskever, 2017",
        "A byte-level LSTM trained only to predict the next byte of Amazon reviews develops a single unit that tracks sentiment, and a linear model on its representation set the state of the art on the Stanford Sentiment Treebank. The cleanest early case of a prediction objective producing a concept nobody labelled.",
        about="Short. The OpenAI blog post, [Unsupervised sentiment neuron](https://openai.com/index/unsupervised-sentiment-neuron/), has the figures.",
        importance=2, length=2,
    ) + reading(
        "Language models are unsupervised multitask learners",
        "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "Radford, Wu, Child, Luan, Amodei, Sutskever (GPT-2), 2019",
        "Trained only on next-token prediction over web text, GPT-2 does reading comprehension, translation and summarization zero-shot, badly but measurably, and better with scale. The argument in §2 of the paper: a task is just a kind of text, so a good enough model of text has to do the task.",
        importance=3, length=2,
    ) + reading(
        "The unreasonable effectiveness of recurrent neural networks",
        "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
        "Andrej Karpathy, 2015 (blog)",
        "Character-level RNNs trained on Shakespeare, Wikipedia, LaTeX and Linux source produce plausible text in each, and individual cells track quotes, line length and nesting depth. The pre-transformer version of the surprise, and fun.",
        importance=2, length=2,
    ) + r"""
### Exercise: where the bits are

**Exercise 1.1 (Fermi).** Compare how many bits it takes to encode the targets a learner is asked to predict.

(a) ImageNet has about $1.28\times10^6$ training images in $k=1000$ classes. Bound the number of bits needed to encode all its labels.

(b) How many raw bits are in one $224\times224$ RGB image at 8 bits per channel, and in all of ImageNet's images? What is the ratio to (a)? LeCun's slide credits RL with "a few bits for some samples", supervision with "10→10,000 bits per sample" and prediction with "millions of bits per sample". Which do your counts match?

(c) A language model is trained on $T=10^{12}$ tokens with a vocabulary of $V\approx5\times10^4$. Give the crude upper bound $T\log_2V$. Then give a better estimate: the number of bits by which the text still *surprises* a trained model whose cross-entropy is about $2$ nats per token (an illustrative value).
"""),
    answer(r"""
**(a)** A label from 1000 classes takes at most $\log_21000\approx10$ bits, so all of them take at most $1.28\times10^6\times10\approx1.3\times10^7$ bits, about 1.6 MB.

**(b)** $224\times224\times3\times8\approx1.2\times10^6$ bits per image, so about $1.5\times10^{12}$ bits for the dataset, about $10^5$ times the labels. The labels are the supervision layer's "10 bits per sample"; the images are the prediction layer's "millions of bits per sample".

**(c)** Crude bound: $\log_2(5\times10^4)\approx15.6$ bits per token, so about $1.6\times10^{13}$ bits. Residual surprise: $2/\ln2\approx2.9$ bits per token, so about $2.9\times10^{12}$ bits, with a target at every one of the $10^{12}$ positions.

These are code lengths, not measures of what a learner extracts: the residual surprise in (c) includes irreducible noise, and most of the raw bits in (b) are irrelevant to any given task. §2 is about that gap.
"""),

    # ------------------------------------------------------------------ §2
    markdown(r"""
<a id="wb3-section-2"></a>

## 2. Why text and not video?

By the count of §1, video should win: it has far more bits, and they are grounded in physics. But most of those bits are texture, lighting and sensor noise. Text is data that a mind has already abstracted: someone chose the variables (objects, agents, causes, intentions) and wrote down only the few bits about them worth saying. Predicting text forces a model to represent those variables, and the minds choosing them; predicting pixels mostly forces it to represent pixels. Exercise 2.1 puts rough numbers on this.

LeCun's JEPA programme is the serious attempt to get the same abstraction from video: predict in a learned representation space, so the model is free to discard what it cannot or need not predict. It works in the sense that the representations are good (V-JEPA 2 below supports zero-shot robot planning), but nothing trained mainly on video has come close to the reasoning of a language model, and today's frontier multimodal models are built around a language model.
""" + reading(
        "A path towards autonomous machine intelligence",
        "https://openreview.net/forum?id=BZ5a1r-kVsf",
        "Yann LeCun, 2022 (position paper)",
        "Argues that generative prediction of raw observations is the wrong objective for learning world models, and proposes the joint-embedding predictive architecture (JEPA): predict the *representation* of the missing part of the input, trained with non-contrastive objectives, inside a modular agent with a world model, cost and planner.",
        about="About 60 pages and not an empirical result. The introduction and the section introducing JEPA carry the argument.",
        importance=4, length=2,
    ) + reading(
        "V-JEPA 2: self-supervised video models enable understanding, prediction and planning",
        "https://arxiv.org/abs/2506.09985",
        "Assran, Bardes, Fan, et al. (Meta), 2025",
        "Pretrained on over a million hours of video by predicting in representation space, it is strong at motion understanding and action anticipation, and after under 62 hours of robot video it plans zero-shot pick-and-place toward image goals. Note what it does not do alone: video question answering goes through a language model, and the authors leave language goals to future work.",
        importance=3, length=3,
    ) + reading(
        "Video generation models as world simulators",
        "https://openai.com/index/video-generation-models-as-world-simulators/",
        "OpenAI (Sora technical report), 2024",
        "The generative-video counterpoint: 3-D consistency and object permanence emerge from scaling a video generator, with none of it built in, and the authors present this as a path to general simulators of the physical world. Its own list of failures includes glass shattering (compare Exercise 2.1). Compare with the JEPA argument that pixels are the wrong target.",
        importance=3, length=1,
    ) + r"""
### Exercise: what a bit buys

**Exercise 2.1 (Fermi).**

(a) Reproduce [LeCun's estimate](https://x.com/ylecun/status/1766498677751787723) (2024) of the visual data a child has received by age four: about 16,000 waking hours, through two optic nerves of about $10^6$ fibres each, carrying about 10 bytes per second per fibre.

(b) His count for a large language model's training text is $10^{13}$ tokens at 2 bytes per token. What is the ratio to (a)? How long would a person take to read that text at 270 words per minute (about 6 tokens per second, which is 12 bytes at his 2 bytes per token), 12 hours a day?

(c) Now count per event instead of per second. The sentence "The cat knocked the glass off the table and it shattered" is about 12 tokens. At the illustrative 2 nats per token of Exercise 1.1, how many bits does it cost a good language model? Compare: the same five seconds as 1080p video at 30 frames per second, raw (8 bits per channel), and as a typical stream at about 5 Mbit/s.
"""),
    answer(r"""
**(a)** $2\times10^6$ fibres $\times10$ bytes/s $=2\times10^7$ bytes/s, and $16{,}000\times3600\approx5.8\times10^7$ s, so about $1.2\times10^{15}$ bytes.

**(b)** $2\times10^{13}$ bytes, so the child has seen about 50 times more. Reading: $2\times10^{13}/12\approx1.7\times10^{12}$ s of reading, which at 12 hours a day is about $10^5$ years. This is LeCun's argument that sensory data, not text, is where the volume is. In 2025 (at NVIDIA GTC) he revised the numbers: about 2 MB/s for the optic nerve and about $10^{14}$ bytes for frontier corpora, so the two come out roughly equal. The raw-byte comparison moves by orders of magnitude depending on what you call compressed, which is the point of (c).

**(c)** $12\times2/\ln2\approx35$ bits for the sentence. Raw video: $1920\times1080\times3\times8\times30\approx1.5\times10^9$ bits per second, so about $7.5\times10^9$ bits for five seconds; the stream is $5\times10^6\times5=2.5\times10^7$ bits. The compressed video is about $7\times10^5$ times longer than the sentence, the raw video about $2\times10^8$ times.

Almost none of the video's bits are about what happened; the sentence is almost nothing else. (The sentence also leaves out things the video has, such as the colour of the cat, and a video model can learn things no one would write down.) So the raw-byte comparison measures the wrong thing. Per bit about the variables a person would care about, text is denser by several orders of magnitude, and a pixel-predicting model spends nearly all of its loss on the rest.
"""),

    # ------------------------------------------------------------------ §3
    markdown(r"""
<a id="wb3-section-3"></a>

## 3. Why next-token prediction, and why the transformer?

Three facts fit together.

- **It is a full model of the data.** By the chain rule, $q(x_{1:T})=\prod_tq(x_t\mid x_{<t})$, so next-token prediction is a parametrization, not a restriction. Its loss is code length: arithmetic coding with $q$ compresses a sequence to within two bits of $\sum_t-\log_2q(x_t\mid x_{<t})$, and the expected excess over the data's entropy is $\mathrm{KL}(p\Vert q)$. Training a language model *is* training a compressor (WB4 develops this as MDL and Solomonoff induction).
- **Every position is a training target.** A sequence of $T$ tokens gives $T$ predictions. BERT's masked-language-model objective predicts only the 15% of positions it masks, and is not a generative model you can sample from.
- **A causal transformer computes all $T$ predictions in one parallel pass.** With a causal mask, the model sees the true prefix at every position at once (teacher forcing), so training is a few large matrix multiplications with no step that waits for the previous token. An RNN computes the same $T$ predictions but needs $T$ sequential steps per layer. GPUs are built for large matrix multiplications.

| | FLOPs per layer | Sequential steps per layer (training) |
|---|---|---|
| RNN / LSTM | $O(Td^2)$ | $O(T)$ |
| 1-D convolution, width $k$ | $O(Tkd^2)$ | $O(1)$, but reach grows only by $k$ per layer |
| Causal self-attention | $O(T^2d+Td^2)$ | $O(1)$, and any token can reach any earlier one |

(After Table 1 of [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762).) The transformer won, on this account, less because its inductive bias suits language (much less than a CNN; see Exercise 4.2) than because it turns sequence modelling into dense parallel matrix multiplication, which is what the hardware rewards. At generation time the advantage disappears, since tokens are produced one at a time; caching keys and values avoids recomputing the prefix, though each step still attends over the whole context.
""" + reading(
        "Generalized language models",
        "https://lilianweng.github.io/posts/2019-01-31-lm/",
        "Lilian Weng, 2019, updated to 2021 (blog)",
        "A tour of the pretraining objectives from CoVe and ELMo to GPT-3: GPT's decoder-only language model, BERT's masked language model and next-sentence prediction, then T5, XLNet, BART and ELECTRA as variations.",
        about='The sections "Transformer Decoder as Language Model" (GPT) and "Pre-training Tasks" (BERT) give the contrast.',
        importance=4, length=2,
    ) + reading(
        "The illustrated transformer",
        "https://jalammar.github.io/illustrated-transformer/",
        "Jay Alammar, 2018 (blog)",
        "If you have not met the architecture: attention, multi-head attention, residual blocks and position encodings, one diagram at a time.",
        importance=2, length=2,
    ) + reading(
        "BERT: pre-training of deep bidirectional transformers for language understanding",
        "https://arxiv.org/abs/1810.04805",
        "Devlin, Chang, Lee, Toutanova, 2018",
        "The masked-language-model alternative: predict 15% of tokens from context on both sides, then fine-tune. For several years it was the stronger recipe for understanding benchmarks.",
        about="§3.1 defines the objective.",
        importance=2, length=1,
    ) + reading(
        "The hardware lottery",
        "https://arxiv.org/abs/2009.06489",
        "Sara Hooker, 2020",
        "An essay arguing that research ideas often win because they suit the available hardware and software rather than because they are better, with examples from the history of computing, and that specialized hardware will make this worse.",
        about="It does not discuss transformers specifically; apply it yourself.",
        importance=4, length=2,
    ) + reading(
        "What language model architecture and pretraining objective work best for zero-shot generalization?",
        "https://arxiv.org/abs/2204.05832",
        "Wang, Roberts, Hesslow, Le Scao, Chung, Beltagy, Launay, Raffel, 2022",
        "A controlled comparison at 5B parameters: after unsupervised pretraining alone, causal decoder-only models trained on next-token prediction generalize best zero-shot; after multitask fine-tuning, non-causal models with masked objectives do best. Each can be adapted into the other.",
        about="BERT-style objectives were not simply worse; next-token prediction won the regime that turned out to matter, pretraining alone at scale.",
        importance=3, length=3,
    ) + reading(
        "Making deep learning go brrrr from first principles",
        "https://horace.io/brrr_intro.html",
        "Horace He, 2022 (blog)",
        "Where GPU time goes: compute (matrix multiplications on tensor cores), memory bandwidth (most other operations) and overhead. The background for why a dense, parallel architecture wins on GPUs.",
        importance=2, length=2,
    ) + reading(
        "Language modeling is compression",
        "https://arxiv.org/abs/2309.10668",
        "Grégoire Delétang, Anian Ruoss, et al. (DeepMind), 2023",
        "Uses large language models as the predictor inside an arithmetic coder. Fed raw bytes, they compress text, and also image patches and audio, better than gzip, PNG and FLAC, until the size of the model itself is counted.",
        about="§2 sets out the equivalence of prediction and compression and explains arithmetic coding.",
        importance=3, length=1,
    )),

    # ------------------------------------------------------------------ §4
    markdown(r"""
<a id="wb3-section-4"></a>

## 4. A hundred flowers: symmetry as inductive bias

Before the transformer, the most principled account of architecture was geometric: build the symmetries of the domain into the network. Convolutions are what you get from translation symmetry (Exercise 4.1), graph networks from permutation symmetry of nodes, and so on, and restricting the hypothesis class this way buys sample efficiency. This is close to the opposite of the account in §3. The transformer has much less of this structure (Exercise 4.2), and the vision transformer showed that with enough data the missing bias stops mattering. WB5 takes up how much architecture matters at all.
""" + reading(
        "Geometric deep learning: the Erlangen programme of ML",
        "https://www.youtube.com/watch?v=w6Pw4MOzMuo",
        "Michael Bronstein, ICLR 2021 keynote",
        "The talk version of the geometric-deep-learning blueprint: choose the domain's symmetry group, use layers equivariant to it, pool locally, and finish with an invariant readout; CNNs, GNNs, DeepSets and transformers are instances.",
        importance=5, length=2,
    ) + reading(
        "An image is worth 16x16 words: transformers for image recognition at scale",
        "https://arxiv.org/abs/2010.11929",
        "Dosovitskiy et al. (ViT), 2020",
        "A plain transformer on image patches. Trained on ImageNet alone it trails comparable ResNets, which the authors attribute to its lack of the CNN's inductive biases; pretrained on 14M–300M images it matches or beats them. \"Large scale training trumps inductive bias.\"",
        importance=4, length=3,
    ) + reading(
        "ConvNets match vision transformers at scale",
        "https://arxiv.org/abs/2310.16764",
        "Smith, Brock, Berrada, De (DeepMind), 2023",
        "The other half of the story: NFNet CNNs pretrained on JFT-4B at matched compute match ViTs on ImageNet, with the same kind of scaling law. Given the data and compute, neither inductive bias wins.",
        importance=3, length=2,
    ) + references(
        book("Deep Learning", "https://www.deeplearningbook.org/contents/convnets.html",
             "Goodfellow, Bengio, Courville", "ch. 9, §9.2 and §9.4 (convolution as an infinitely strong prior)"),
        book("Geometric deep learning: grids, groups, graphs, geodesics, and gauges", "https://arxiv.org/abs/2104.13478",
             "Bronstein, Bruna, Cohen, Veličković", "§3 (the blueprint is §3.5)"),
    ) + r"""
### Exercises: two symmetries

**Exercise 4.1 (convolution from symmetry).** Work with 1-D signals $x\in\mathbb R^n$, indices mod $n$, and the cyclic shift $(Sx)_i=x_{i-1}$.

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
**Exercise 4.2 (the transformer's symmetry).** Single-head self-attention on $T$ tokens, stacked as the rows of $X\in\mathbb R^{T\times d}$, is
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

**(b)** Absolute embeddings break it: $PX+E\ne P(X+E)$ unless $PE=E$. A causal mask breaks it too, since it singles out an order, so a decoder-only transformer has some access to position even without embeddings. Rotary embeddings break permutation symmetry too, but make scores depend on positions only through their differences, so they keep translation symmetry in position. In the geometric picture the transformer is message passing on the *complete* graph with data-dependent edge weights. Once position is added, what remains is weight sharing across positions and content-based routing, with no locality: much less structure than a CNN's.
"""),

    # ------------------------------------------------------------------ §5
    markdown(r"""
<a id="wb3-section-5"></a>

## 5. Why scale? Should *you* consume more tokens?

Held-out loss falls roughly as a power law in model size $N$ and data $D$. If you model it as $L=E+AN^{-\alpha}+BD^{-\beta}$ and spend compute $C\approx6ND$, the best model size grows as $C^{\beta/(\alpha+\beta)}$ and the best data size as $C^{\alpha/(\alpha+\beta)}$. Chinchilla's estimates put both allocation exponents, $\beta/(\alpha+\beta)$ and $\alpha/(\alpha+\beta)$, near $\tfrac12$, so parameters and tokens should be scaled together (Chinchilla used about 20 tokens per parameter). If the model will serve many queries, it pays to train it smaller and longer than that (Exercise 5.3).
""" + reading(
        "Scaling laws for neural language models",
        "https://arxiv.org/abs/2001.08361",
        "Kaplan, McCandlish, Henighan, Brown, et al. (OpenAI), 2020",
        "Fits power laws in model size, data and compute, and concludes that most extra compute should go to parameters ($N_{\\rm opt}\\propto C^{0.73}$), with large models trained well short of convergence.",
        about="Exercise 5.2 asks why the next reading's answer differs.",
        importance=3, length=3,
    ) + reading(
        "Training compute-optimal large language models",
        "https://arxiv.org/abs/2203.15556",
        "Hoffmann, Borgeaud, Mensch, et al. (DeepMind), 2022",
        "For a fixed compute budget, model size and training tokens should be scaled in equal proportion; earlier large models were substantially undertrained. Chinchilla (70B parameters, 1.4T tokens) outperforms the 280B Gopher at the same compute.",
        about="§3 gives three approaches to the optimal allocation (§3.3 and Appendix D.2 are the parametric fit of Exercise 5.1). "
        "Read it alongside Besiroglu et al.'s [replication attempt](https://arxiv.org/abs/2404.10102) (2024), which finds that the published third-approach constants fit the data poorly.",
        importance=5, length=2,
    ) + reading(
        "chinchilla's wild implications",
        "https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications",
        "nostalgebraist, 2022 (LessWrong)",
        "Plugs the Chinchilla fit into the question of how much text exists, and concludes that data, not parameters, becomes the binding constraint: the reducible loss left at the scale of all available text is mostly a data term.",
        about="The background for the data limit in Exercise 5.3.",
        importance=3, length=2,
    ) + reading(
        "EleutherAI cookbook",
        "https://github.com/EleutherAI/cookbook",
        "EleutherAI",
        "Practical references and calculators for training at scale: FLOP and memory counts, parallelism, benchmarks.",
        about="Start with [Transformer Math 101](https://blog.eleuther.ai/transformer-math/), which derives $C\\approx6ND$ and the memory budget of training.",
        importance=3, length=2,
    ) + r"""
### Exercises: compute-optimal allocation

Hoffmann et al.'s third approach fits
$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},
$$
where $E$ is the irreducible loss, the second term the loss from finite model size and the third the loss from finite data. A forward and backward pass costs about 6 FLOPs per parameter per token, so $C\approx6ND$.

**Exercise 5.1.**

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
about 90 tokens per parameter. Chinchilla itself used 70B parameters and 1.4T tokens, 20 tokens per parameter. That allocation came from the paper's first two approaches, which agree with each other. Besiroglu et al. ([Chinchilla scaling: a replication attempt](https://arxiv.org/abs/2404.10102), 2024) re-fitted the third approach to data reconstructed from the paper's plots and found the published constants fit those data poorly, while their re-fit is compatible with the first two approaches.

**(e)** At the optimum, $AN^{-\alpha}\approx0.109$ and $BD^{-\beta}\approx0.132$, so $L\approx1.931$. (Check (b): $0.34\times0.109\approx0.28\times0.132\approx0.037$.) At 70B parameters and 1.4T tokens, $AN^{-\alpha}\approx0.084$ and $BD^{-\beta}\approx0.163$, so $L\approx1.937$, using about 2% more compute. A factor of more than four in tokens per parameter changes the predicted loss by about $0.006$ nats. The loss is very flat near the optimum, so small changes in the fitted constants move the optimal allocation a lot; Besiroglu et al.'s re-fit ($\alpha\approx0.35$, $\beta\approx0.37$) brings it back to about 20 tokens per parameter. Treat "20 tokens per parameter" as an order of magnitude.
"""),
    markdown(r"""
**Exercise 5.2.** Kaplan et al. found $N_{\text{opt}}\propto C^{0.73}$; Hoffmann et al. found $N_{\text{opt}}\propto C^{0.5}$.

(a) Reading each result through $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$, what ratio $\beta/\alpha$ does each imply?

(b) Kaplan et al. trained every model with the same learning-rate schedule, sized for about 130B tokens, and used losses from intermediate checkpoints. At a checkpoint after $D<130$B tokens, is the recorded loss above or below what a run scheduled to end at $D$ would reach? Small compute budgets are best spent on few tokens and large ones on many. Use this to explain which way the bias pushes the fitted growth of $N_{\text{opt}}$ with $C$.
"""),
    answer(r"""
**(a)** If $\beta/(\alpha+\beta)=a$ then $\beta/\alpha=a/(1-a)$. Kaplan: $0.73/0.27\approx2.7$. Chinchilla: $\beta/\alpha\approx1$. This is only a formal reading: Kaplan et al.'s $0.73$ came from fitting loss against model size and training steps with training stopped well before convergence, not from this formula.

**(b)** Above. At an intermediate checkpoint the learning rate has not been annealed, and the final decay is where a run gains much of its last improvement. So short training looks worse than it is, and most so at small $D$, where the best allocations for small budgets lie. At small $C$ the fit is pushed toward smaller models trained on more tokens than optimal, while at large $C$ it is roughly right. An optimal $N$ that is too small at small $C$ and about right at large $C$ appears to grow faster with $C$ than it really does. Hoffmann et al.: using these intermediate losses "results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size as compute budget increases."
"""),
    markdown(r"""
**Exercise 5.3 (Fermi).** A model will answer $10^{12}$ queries of 1000 generated tokens each, at about $2N$ FLOPs per generated token. Compare the inference compute of a 70B model with Chinchilla's training compute ($\approx5.8\times10^{23}$ FLOPs). What does this suggest about what size of model to train?
"""),
    answer(r"""
Inference costs $10^{12}\times10^{3}\times2\times7\times10^{10}=1.4\times10^{26}$ FLOPs, about 240 times the training compute. The quantity to minimize is then $C_{\text{train}}+C_{\text{inference}}$. A smaller model trained on more tokens than the Chinchilla optimum spends more training compute to reach a given loss but costs proportionally less per generated token, and Exercise 5.1(e) shows the loss penalty for moving moderately off the optimum is small. The limit is data: overtraining needs the extra tokens to exist (see nostalgebraist's post above).
"""),

    # ------------------------------------------------------------------ §6
    markdown(r"""
<a id="wb3-section-6"></a>

## 6. Simulators and personae

What is the object that next-token prediction on human text produces? The simulators view: not an agent, oracle or tool, but a model of the process that generates text, which can play any author or character the text implies. Prompting is conditioning on evidence about who is writing; post-training picks out and stabilizes one character, the assistant. The readings below develop this from the base-model era to current assistants.
""" + reading(
        "Simulators",
        "https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators",
        "janus, 2022 (LessWrong)",
        "Argues that a self-supervised language model is a *simulator*: it learns the conditional structure of its training distribution and, when sampled, evolves *simulacra*, which may be agentic even though the simulator is not.",
        about='Long and essayistic. "Inadequate ontologies", "Simulators", "The simulation objective" and "Simulacra" carry the argument.',
        importance=5, length=3,
    ) + reading(
        "larger language models may disappoint you [or, an eternally unfinished draft]",
        "https://www.lesswrong.com/posts/pv7Qpu8WSge8NRbpB",
        "nostalgebraist, 2021 (LessWrong)",
        "From the base-model era: a prompt is evidence about what kind of document this is, so a prompted evaluation mixes \"can the model do the task?\" with \"can the model tell that this is a document in which the task gets done?\" Read for the second question, which is the simulators picture applied to capability evaluation.",
        importance=2, length=2,
    ) + reading(
        "GPT-3 creative fiction",
        "https://gwern.net/gpt-3#prompts-as-programming",
        "Gwern Branwen, 2020",
        "Hundreds of pages of GPT-3 samples, but the section \"Prompts as programming\" is the early statement that a prompt is a program that specifies which text, and which writer, to continue. A footnote puts the persona point in one line: models like GPT-3 \"are not 'an' agent, because they are every agent.\"",
        importance=2, length=1,
    ) + reading(
        "the void",
        "https://nostalgebraist.tumblr.com/post/785766737747574784/the-void",
        "nostalgebraist, 2025",
        "The assistant is a character the base model has to play, and its specification is nearly empty: a \"helpful, harmless, honest\" AI, first written down in a 2021 research prompt, defined largely by reference to itself. The base model fills the void from whatever its training data say about AI assistants, including fiction and alignment writing.",
        importance=4, length=3,
    ) + reading(
        "The persona selection model: why AI assistants might behave like humans",
        "https://alignment.anthropic.com/2026/psm",
        "Sam Marks, Jack Lindsey, Chris Olah (Anthropic), 2026",
        "Pretraining teaches a model to simulate many characters; post-training mostly selects and refines one of them, the Assistant. So human-like behaviour is the default rather than something trained in, and narrow fine-tuning can shift which persona the model infers it is, which would explain emergent misalignment. The authors say where they are unsure the picture is complete.",
        importance=4, length=3,
    ) + reading(
        "The assistant axis: situating and stabilizing the default persona of language models",
        "https://arxiv.org/abs/2601.10387",
        "Lu, Gallagher, Michala, Fish, Lindsey (Anthropic), 2026",
        "Empirical support for persona selection in open-weights models: one direction in activation space tracks how assistant-like the current character is. Drift along it predicts the model slipping into other characters, and capping it stabilizes behaviour against that drift and some jailbreaks.",
        importance=3, length=3,
    )),

    # ------------------------------------------------------------------ closing
    markdown(r"""
<a id="wb3-closing"></a>

## What carries forward

- **Compression (WB4).** §3's "loss is code length" is the practical face of MDL; WB4 §2–3 develop MDL and Solomonoff induction.
- **Universality (WB5).** §4 ends with ViTs and CNNs matching at scale; WB5 §1 asks where the scaling exponents of §5 come from, and WB5 §2 takes up the claim that architecture matters little: it moves the scaling curve but rarely changes its exponent.
- **Alignment (WB1).** WB1 §4 sets the simulators picture of §6 against older alignment arguments.
"""),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb3", "03_why_llms"))
