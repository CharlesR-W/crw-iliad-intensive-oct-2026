# WB6 source-scope check

Root reviewer, 2026-10-08. These checks concern primary-source abstracts and bibliographic destinations, not a replication or full-paper audit. Editorial revision/re-check is recorded separately.

The arXiv destinations and their abstracts were read for the following existing readings:

- [Layer by Layer](https://arxiv.org/abs/2502.02013): intermediate representations improve downstream embedding performance in the tested tasks and architectures. Avoid an unconditional claim about the most useful layer for every task.
- [SVCCA](https://arxiv.org/abs/1706.05806): reports bottom-up convergence and effective low dimensionality. The abstract's affine-invariance wording should not be generalized to invariance of an arbitrary preceding variance-truncation step.
- [Model stitching](https://arxiv.org/abs/2106.07682): the abstract's supervised/self-supervised interchangeability result concerns networks of the same architecture. Do not silently broaden this particular evidence to arbitrary architectures.
- [Critical learning periods](https://arxiv.org/abs/1711.08856): early low-level stimulus deficits can persist through subsequent training. This is an experimental persistence claim, not a theorem that no later intervention can repair the deficit.
- [Are All Layers Created Equal?](https://arxiv.org/abs/1902.01996): heterogeneous sensitivity to resetting layers; the paper cautions that architecture matters. A finite reset that preserves performance is not a proof of local Hessian flatness along the path.
- [Adversarial examples and time-limited humans](https://arxiv.org/abs/1802.08195) and [non-robust features](https://arxiv.org/abs/1905.02175): links/titles resolve; keep their separate experiments separate.
- [Distillation](https://arxiv.org/abs/1503.02531), [lottery tickets](https://arxiv.org/abs/1803.03635), and [linear mode connectivity](https://arxiv.org/abs/1912.05671): links/titles resolve. The last paper reports early stability in tested vision models; do not turn linear connectivity into a claim of a unique basin under every notion of basin.
- [Tiny gradient subspace](https://arxiv.org/abs/1812.04754): a measured concentration in top Hessian eigendirections, not a general theorem that only those directions matter for learning.
- [Hessian eigenvalue density](https://arxiv.org/abs/1901.10159): outliers and corresponding gradient concentration are prominent in the tested non-batch-normalized networks and largely absent with batch normalization.
- [Sparse RL subnetworks](https://arxiv.org/abs/2505.11711): the source explicitly distinguishes sparse updates from nearly full-rank matrix updates. Its 5–30% numbers refer to the tested models/algorithms and measurement definition.
- [RL's Razor](https://arxiv.org/abs/2509.04259): reports less forgetting and smaller distribution shift for on-policy RL than SFT in tested settings; this is not a universal no-new-capabilities theorem.
- [Physics of Language Models, Part 1](https://arxiv.org/abs/2305.13673): synthetic CFG experiments, hidden structure and attention patterns resembling dynamic programming. Keep the synthetic task scope visible.

These constraints are for checking the final short reading descriptions. No new student-facing discussion rubric is implied.
