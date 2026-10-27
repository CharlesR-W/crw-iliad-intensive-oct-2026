#!/usr/bin/env python3
"""Loose reading collection for October 29; ordering beyond the opener is provisional."""
from nbkit import markdown, reading, answer, run

CELLS = [markdown(r"""
# Ambitious Mechinterp and Activation Geometry — reading collection

**Iliad Intensive · Thursday 29 October 2026**

Readings to accompany Eric Michaud and Brianna Grado-White's lectures. Pick what interests you; this is more than a day's reading. These are loosely grouped starting points, not a prescribed sequence. Importance scores prioritise this day’s reading: 5 marks the four anchor readings; 4 a recommended next choice; 3 useful depth; 2 an optional extension; 1 a specialist rabbit hole. Length is scored separately. You are not expected to read every high-rated item.

**Start with Charles's “What if not Circuits?”, section I.** The rest of that essay is optional. As you read the papers below, ask what each takes a feature or circuit to be, what evidence would establish its existence, and what its methods actually measure.
""")]

def section(title, entries, optional_from=None):
    def render(items):
        return ''.join(reading(t,u,a,n,importance=i,length=l) for t,u,a,n,i,l in items)
    body = render(entries if optional_from is None else entries[:optional_from])
    if optional_from is not None:
        body += '\n\n<details>\n<summary>Optional: SAE variants (Gated, Top-K, JumpReLU, Matryoshka)</summary>\n\n'
        body += render(entries[optional_from:]) + '\n</details>\n'
    CELLS.append(markdown('## '+title+'\n\n'+body))

section('Framing: what are we trying to explain?', [
('What if not Circuits?', 'https://www.lesswrong.com/posts/mMERyrvEJ4xbiozie/what-if-not-circuits', 'Charles Renshaw-Whitman, 2026', 'First reading: section I, “What Might We Mean By Circuits?” Distinguish assumptions about circuits from assumptions about representations; treat the essay as a perspective to examine, not a conclusion required by this notebook.',5,2),
('A Pragmatic Vision for Interpretability','https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-inter','Neel Nanda et al., 2025','The explicit turn toward task-grounded, empirically evaluated interpretability; compare its success criteria with the ambition of reverse-engineering a model.',4,3),
])
section('Representations and Toy Models of Superposition', [
('Toy Models of Superposition','https://transformer-circuits.pub/2022/toy_model/index.html','Elhage et al., 2022','The synthetic setting where features are known: investigate when sparse features share dimensions, how interference is filtered, and which conclusions depend on the toy setup.',5,4),
('Distributed Representations: Composition & Superposition','https://transformer-circuits.pub/2023/superposition-composition/index.html','Chris Olah, 2023','Distinguishes compositional codes, superposition, and mere changes of basis. This is the likely intended “representation types” note, and is also linked from Charles’s opening essay.',4,2),
])
CELLS += [markdown(r"""
### Exercises: interference and what a probe proves

In the tied-weight toy model, $x\in\mathbb R^n$, $W\in\mathbb R^{m\times n}$, $m<n$:
$$h=Wx,\qquad \hat x=\operatorname{ReLU}(W^T h+b),\qquad
\mathcal L=\mathbb E_x\sum_i I_i(x_i-\hat x_i)^2.$$
Here $I_i$ are feature-importance weights. These are synthetic features supplied by the experimenter.

1. Writing $w_i$ for column $i$ of $W$, expand the preactivation of reconstructed feature $i$. Identify the self-term and interference. Why does sparsity help without making interference vanish?
2. Encode each colour–shape pair with a separate one-hot coordinate. Construct a perfect linear probe for “red”. Does this prove that a dedicated red coordinate is used by the model's computation?
"""), answer(r"""
1. $\hat x_i=\operatorname{ReLU}(\|w_i\|^2x_i+\sum_{j\ne i}w_i^Tw_jx_j+b_i)$. Sparsity reduces typical coactivation; it does not force the cross-inner-products to zero. Bias and ReLU can reject weak interference, at the cost of discarding some signal. A claim about performance also needs the distribution of feature magnitudes and co-occurrences.
2. Assign probe weight 1 to every red–shape coordinate and 0 elsewhere. It predicts red perfectly on the one-hot inputs, although the code has no separate red coordinate. Decodability alone neither identifies the model's chosen factorisation nor proves downstream causal use.
""")]
section('From dictionary learning to scaled features', [
('Towards Monosemanticity: Decomposing Language Models With Dictionary Learning','https://transformer-circuits.pub/2023/monosemantic-features/index.html','Bricken et al., 2023','The bridge from toy-model features to learned dictionaries in a small language model.',3,4),
('Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet','https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html','Templeton et al., 2024','Read the extraction method, examples and limitations together; feature activation, feature naming and steering provide different kinds of evidence.',5,4),
('Improving Dictionary Learning with Gated Sparse Autoencoders','https://arxiv.org/abs/2404.16014','Rajamanoharan et al., 2024','Separates feature selection from magnitude estimation to reduce shrinkage.',1,3),
('Scaling and evaluating sparse autoencoders','https://arxiv.org/abs/2406.04093','Gao et al., 2024','Top-K sparsity, scaling and evaluation; compare reconstruction with the separate tests of feature quality.',2,3),
('Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders','https://arxiv.org/abs/2407.14435','Rajamanoharan et al., 2024','Thresholded activations and an L0 objective, trained with straight-through estimators.',1,3),
('Learning Multi-Level Features with Matryoshka Sparse Autoencoders','https://arxiv.org/abs/2503.17547','Bussmann et al., 2025','Nested dictionaries as an attempt to capture features at multiple levels of abstraction.',2,3),
], optional_from=2)
CELLS += [markdown(r"""
### Exercises: what the sparsity penalty buys

A schematic vanilla SAE has $z=\operatorname{ReLU}(W_e(x-b_d)+b_e)$, $\hat x=W_dz+b_d$, and objective
$$\mathbb E[\|x-\hat x\|_2^2+\lambda\|z\|_1],$$
with decoder-column scale controlled. This schematic is not the objective of every SAE variant.

1. For a unit decoder direction $d$ and a fixed residual $r$, minimise $\|r-ad\|^2+\lambda a$ over $a\ge0$. What bias does L1 introduce?
2. Without controlling decoder scale, replace $d$ by $cd$ and $a$ by $a/c$. What happens to reconstruction and the penalty?
3. Propose a comparison in which better reconstruction would not by itself establish better interpretability.
"""), answer(r"""
1. Differentiating yields $a^*=\max(0,d^Tr-\lambda/2)$. The active coefficient is shrunk relative to the nonnegative least-squares estimate. The factor of two changes if reconstruction loss includes a factor $1/2$.
2. Reconstruction is unchanged while the penalty decreases by $1/c$: an uncontrolled scaling degeneracy.
3. Compare methods at matched sparsity and dictionary size, then separately measure held-out concept detection, intervention specificity and downstream loss recovery. A dictionary can improve reconstruction by encoding nuisance variation without improving the chosen task. None of these measurements alone certifies a complete ontology.
""")]
section('Circuit tracing and circuit discovery', [
('Circuit Tracing: Revealing Computational Graphs in Language Models','https://transformer-circuits.pub/2025/attribution-graphs/methods.html','Ameisen et al., 2025','Cross-layer transcoders and attribution graphs: read the replacement-model construction and its limitations before interpreting edges as explanations of the original model.',4,4),
('On the Biology of a Large Language Model','https://transformer-circuits.pub/2025/attribution-graphs/biology.html','Lindsey et al., 2025','Choose one case study, inspect its graph and interventions, and separate observations from the proposed computational story. Paired with the methods paper above; this is our working interpretation of “CTC/biology”.',4,4),
('Towards Automated Circuit Discovery for Mechanistic Interpretability','https://arxiv.org/abs/2304.14997','Conmy et al., 2023','ACDC searches for task-relevant edges using patching; the dataset, graph granularity, metric and corruption scheme specify the discovery problem.',3,3),
])
CELLS += [markdown(r"""
### Exercise: reconstruction is not mechanism equivalence

Construct $f,g:\mathbb R\to\mathbb R$ that agree on a training distribution supported on $[-1,1]$ but disagree after an intervention that sets $x=2$. What additional checks would you want before transferring an attribution graph from a replacement model to the original?
"""), answer(r"""
Take $f(x)=x$ and $g(x)=x+\operatorname{ReLU}(x-1)$. They agree everywhere on the stipulated support, but $f(2)=2$ and $g(2)=3$. Check the original and replacement under the actual interventions, relevant held-out distributions, and matched behavioural metrics; inspect reconstruction-error contributions and mechanisms the replacement leaves fixed. Agreement on ordinary inputs is insufficient to guarantee agreement under interventions.
""")]

section('Causal scrubbing and causal abstraction', [
('Causal Scrubbing: a method for rigorously testing interpretability hypotheses','https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing','Chan et al., Redwood Research, 2022','Read sections 2–4 and limitations: test a specified explanation with constrained resampling. Distinguish the original proposal from the later retrospective comments.',5,3),
('Causal Abstractions of Neural Networks','https://arxiv.org/abs/2106.02997','Geiger et al., 2021','Interchange interventions test whether a neural representation plays the causal role assigned by a higher-level model.',3,3),
('Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations','https://arxiv.org/abs/2303.02536','Geiger et al., 2023; CLeaR 2024','Distributed Alignment Search learns subspaces aligned with hypothesised variables; distinguish fitting the alignment from testing it.',4,3),
('Causal Abstraction for Mechanistic Interpretability','https://jmlr.org/papers/volume26/23-0058/23-0058.pdf','Geiger et al., JMLR 2025','Longer theoretical reference: maps between high- and low-level interventions, unifying several interpretability methods. Read selected sections.',1,5),
('How to use and interpret activation patching','https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching','Heimersheim and Nanda, 2024','Practical companion on corruption, noising versus denoising, redundancy and metric choice.',2,3),
])
CELLS += [markdown(r"""
### Exercises: interchange and redundancy

For an orthogonal rotation $R$, coordinate projector $P$, base state $h_b$ and source state $h_s$, a subspace interchange is
$$h'=R^T[(I-P)Rh_b+PRh_s]=h_b+Q(h_s-h_b),\qquad Q=R^TPR.$$
1. Show that $Q$ is an orthogonal projector and that $(I-Q)h'=(I-Q)h_b$. What would an evaluation on the same pairs used to learn $R$ fail to establish?
2. Let clean inputs be $A=B=1$ and corrupted inputs $A=B=0$. Compare individually restoring one input and individually corrupting one input for $C=A\land B$ and $C=A\lor B$.
"""), answer(r"""
1. $Q^T=Q$ and $Q^2=R^TPRR^TPR=Q$. Hence $(I-Q)Q=0$, proving complement preservation. Fitting an alignment to examples does not establish generalisation to held-out interventions, or rule out competing high-level explanations.
2. For AND, restoring either input alone leaves 0; corrupting either alone destroys the clean output. For OR, restoring either alone gives 1; corrupting either alone leaves 1. The apparent importance of a component depends on intervention direction and redundancy.
""")]
section('Negative results and changes of direction', [
('Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research','https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks','Lewis Smith et al., 2025','The GDM report linked from Charles’s essay: SAE-based approaches underperformed linear probes on the studied harmful-intent OOD detection task. Read the task and baselines before generalising the conclusion.',4,3),
('Activation space interpretability may be doomed','https://www.lesswrong.com/posts/gYfpPbww3wQRaxAFD/activation-space-interpretability-may-be-doomed','Chughtai and Bushnaq, 2025','An argument distinguishing structure visible in activations from mechanisms used in computation; a bridge to parameter decomposition, not a theorem that SAEs never help.',3,2),
('How useful is mechanistic interpretability?','https://www.lesswrong.com/posts/tEPHGZAb63dfq2v8n/how-useful-is-mechanistic-interpretability','Greenblatt, Nanda and Shlegeris, 2023','A dialogue about faithful explanation and practical impact; compare with the later Pragmatic Vision reading at the top.',2,4),
])
section('Goodfire: feature geometry and parameter decomposition', [
('Understanding sparse autoencoder scaling in the presence of feature manifolds','https://arxiv.org/abs/2509.02565','Eric J. Michaud, Liv Gorton and Tom McGrath, 2025','Especially relevant to the day’s lecturer: dictionary capacity can be spent covering manifolds rather than identifying additional underlying features.',4,3),
('Can SAEs Capture Neural Geometry?','https://www.goodfire.com/research/can-saes-capture-neural-geometry','Bhalla, Fel et al., Goodfire, 2026','How linear dictionary elements can capture, fragment or dilute curved structure; compare the synthetic demonstrations with model examples.',3,3),
('Uncovering Neural Geometry in Vision Models With Block-Sparse Featurizers','https://www.goodfire.com/research/bsf-vision','Fel, Kowal et al., 2026','A move from individual directions to multidimensional subspaces; the empirical setting here is vision models.',2,3),
('Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition','https://arxiv.org/abs/2501.14926','Braun et al., 2025','APD seeks simple, sparsely used parameter components with faithful behaviour, demonstrated in toy systems.',3,4),
('Towards Scalable Parameter Decomposition','https://www.goodfire.com/research/stochastic-param-decomp','Bushnaq et al., Goodfire, 2025','SPD replaces attribution-based selection with stochastic ablation; a bridge from APD to subsequent methods.',2,3),
('Interpreting Language Model Parameters','https://www.goodfire.com/research/interpreting-lm-parameters','Bushnaq et al., Goodfire, 2026','VPD and a 67M-parameter language model. Prioritise the introduction, section 2 and section 7, including “Attribution graphs are not computational graphs” and the adversarial robustness limitations.',4,5),
('Neural Networks Are Already Clean','https://already-clean.goodfire.com/','Nick Cammarata, 2026','Optional contemporary counterpoint: an informal working hypothesis about neuron interpretability, not an established refutation of superposition.',1,4),
])
CELLS += [markdown(r"""
### Exercises: geometry and the choice of parts

1. Let $h(\theta)=(\cos\theta,\sin\theta)$. Distinguish its manifold dimension, linear-span dimension and the number of dictionary directions one might use to approximate it. Why might more SAE latents fail to imply more underlying concepts?
2. For $f(x)=W_2W_1x$ and invertible $R$, show that $\widetilde W_1=RW_1$, $\widetilde W_2=W_2R^{-1}$ preserves the function. Does deleting hidden coordinate $i$ before and after this change of basis generally preserve the same ablated function?
"""), answer(r"""
1. The circle is a one-dimensional manifold with a two-dimensional linear span. A dictionary can use arbitrarily many local directions to describe it. Counting those elements counts a chosen discretisation, not necessarily independent semantic factors.
2. $W_2R^{-1}RW_1=W_2W_1$. Deleting coordinate $i$ in the new basis instead inserts $I-R^{-1}e_ie_i^TR$ between $W_2$ and $W_1$, generally different from $I-e_ie_i^T$. Exact functional or parameter reconstruction does not uniquely pick the parts whose interventions should be meaningful.
""")]
section('Tensor similarity', [
('When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability','https://arxiv.org/abs/2605.15183','Nissen Gonzalez et al., including Logan Riggs Smith, 2026','Weight-based similarity accounting for symmetries in multilinear/tensor models. Read the architecture restrictions: this is not automatically a metric for arbitrary ordinary transformers.',2,4),
])
section('Activation plateaus: Heimersheim and collaborators', [
('Activation Plateaus: Where and How They Emerge','https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge','Matthew Shinkle and Stefan Heimersheim, 2025','Start here: interpolating early activations can yield flat downstream responses separated by sharp transitions. GPT-2 Large experiments trace their gradual sharpening across layers to MLP sensitivity.',4,2),
('[Interim research report] Activation plateaus & sensitive directions in GPT2','https://www.lesswrong.com/posts/LajDyGyiyX8DNNsuF/interim-research-report-activation-plateaus-and-sensitive-1','Stefan Heimersheim and Jake Mendel, 2024','The original perturbation experiments and error-correction motivation. Read the September 2024 correction: confidence in the sensitive-direction comparison was reduced after identifying baseline effects.',3,3),
('Evidence for feature-specific error correction in LLMs','https://arxiv.org/abs/2606.24964','Francisco Ferreira da Silva and Stefan Heimersheim, 2026','Optional follow-up comparing contrastive candidate feature directions with mixtures of directions, connecting plateau geometry to the error-correction hypothesis.',2,3),
])
CELLS += [markdown(r"""
### Exercise: a plateau along which direction?

Let $F$ be the remainder of a network after a chosen activation layer. Along a straight interpolation $h(t)=(1-t)h_A+th_B$, define $y(t)=F(h(t))$. The chain rule gives
$$\frac{dy}{dt}=J_F(h(t))(h_B-h_A).$$
1. Does a small derivative along this path imply a small Jacobian in every direction? Give a counterexample.
2. For residual blocks $h_{\ell+1}=h_\ell+g_\ell(h_\ell)$, derive the recurrence for $v_\ell=dh_\ell/dt$. Why can sensitivity sharpen across layers, and why is multiplying the norms of the MLP Jacobians not the exact calculation?
3. What comparison would help distinguish a learned plateau around real activations from an artefact of the chosen perturbation distribution?
"""), answer(r"""
1. No. For $F(x_1,x_2)=100x_2$, a path parallel to the first coordinate has zero derivative although the Jacobian norm is 100. Flatness along one path need not imply a robust neighbourhood in all directions.
2. $v_{\ell+1}=[I+J_{g_\ell}(h_\ell)]v_\ell$. Ordered Jacobian products can amplify some path segments more than others; sharpening is possible, not guaranteed. Residual identities and the alignment of vectors and singular directions matter. Products of operator norms give bounds, not generally equalities.
3. Compare real and matched synthetic base activations with the same perturbation procedure; control scale, covariance, layer, path and output metric. Examine several directions and both norm-preserving and straight paths. Even a robust difference does not by itself identify discrete features or establish error correction as the unique explanation.
""")]
section('Computation in superposition', [
('Mathematical Models of Computation in Superposition','https://arxiv.org/abs/2408.05451','Hänni, Mendel, Vaintrob and Chan, 2024','Constructive results for sparse computation in narrower networks; distinguish existence from what SGD learns in language models.',4,4),
('Compressed Computation is (probably) not Computation in Superposition','https://arxiv.org/abs/2606.14673','Bhagat et al., including Heimersheim, 2026','A particular toy benchmark’s apparent advantage is explained by a label-mixing term. This is a benchmark critique, not an impossibility theorem.',2,3),
('Compressed Computation under L⁴ Loss is likely Computation in Superposition','https://arxiv.org/abs/2607.04800','Ferreira da Silva and Heimersheim, 2026','Read beside the preceding negative result: a changed loss elicits behaviour explained using sparse codewords and pseudoinverse decoding.',2,3),
])
CELLS += [markdown(r"""
### Exercise: changing which errors matter

Compare residual vectors $(2,0,0,0)$ and $(1,1,1,1)$ under $\sum_i e_i^2$ and $\sum_i e_i^4$. Compute both gradients. Why is a change in training outcome under the new objective insufficient on its own to establish computation in superposition?
"""), answer(r"""
Both squared losses are 4; the fourth-power losses are 16 and 4. The coordinate gradients are $2e_i$ and $4e_i^3$. Fourth-power loss penalises concentrated large errors more strongly. That motivates a different solution but does not identify its mechanism: inspect the learned representation, decoding and intervention evidence.
""")]

if __name__ == '__main__':
    raise SystemExit(run(CELLS, 'ami', 'ami_reading_collection'))
