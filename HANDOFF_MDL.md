# Handoff: Mysteries of Deep Learning workbooks

For whoever (human or model) critiques and revises `MDL_WORKBOOK_SPECS_v1.md`. Please treat yourself kindly while doing this; candour is more useful than agreement, and the instructor wants objections, not polish.

## What this is
A one-day course at the Iliad Intensive (Bay Area run, Oct 5–30 2026; day runs 10:00–18:00 with lunch 12–13). Instructor: CRW. Cohort: grad-level math/physics/TCS, uneven ML practice, learning theory not assumed, Python not assumed. They will have had an SLT (singular learning theory) day shortly before. A sister day, Training Dynamics (also CRW), follows; a Data Attribution day (Louis Jaburi) follows that.

Format: two short orienting lectures (L1 "GOFAI → DL", L2 "DL → LLMs and scale") are the trunk. Six workbooks are the branches; students choose freely and self-pace. Workbooks are markdown-first notebooks in the style of the Physics of Deep Learning day at `/home/crw/Programming/IliadPhysicsLearning` (read its `README.md`, `AGENTS.md`, and `notebooks/second-draft/build_notebook_1.py` for the exact builder conventions: exercise and solution variants from one builder, blank answer cells, hints in `<details>`, claim boundaries, "what carries forward", "further study"). No PLOT-SLOP story chapter this time.

## Files
- `MDL_WORKBOOK_SPECS_v1.md`: the specs. Lectures L1, L2; workbooks WB1–WB6; build notes.
- `CURRICULUM_DRAFT_v1.md` §4: verified reading links (about 130), plus §5 a Codex critique of an earlier timetable-based plan. The spec's workbook-specific links are appended at the end of the spec file.
- `CURRICULUM_TREE_v2.md`: a superseded tree-of-nodes plan; useful only for the "card" convergence idea and suggested paths.
- `Mysteries of Deep Learning`, `Training Dynamics`: CRW's raw topic notes. These are the things CRW is comfortable lecturing on.
- `Iliad B-track - current materials and prereqs.md`: survey of the existing course this replaces, and what can be assumed of the cohort.
- CRW's blog (https://charlesr-w.github.io/crw-blog/) for voice and epistemics: "Power laws are not enough", "What if not circuits", "Selection for selectability", "KL divergence, geometric naturalness, and optimal transport", "Every measurement has a scale".

## Instructor's commitments (do not relitigate; do critique the execution)
- Short lectures, self-guided work, eclecticism. Students may spend the day on one workbook.
- Theory should be the kind a practising ML/alignment researcher uses. Cut theorist habits that practitioners never touch.
- Three estimands kept distinct wherever generalization appears: single-network excess population loss; Bayes-ensemble (posterior-predictive) error; Gibbs (posterior-draw) error. The instructor's view is that the first is what matters for trained networks, and that SLT's RLCT describes the second. This should be visible as content, not as rhetoric: competing accounts get presented symmetrically and the student is asked what evidence discriminates them.
- "Inductive bias" is treated as a thing to mechanize, not a terminal explanation.
- Geometric naturalness (choose a metric that respects the space) is a recurring theme; the GAN→WGAN parable lives in the Training Dynamics day, not here.
- Program synthesis (Furman) is presented as one live hypothesis among others, not as the anchor.

## What a critique should cover, in order
1. **Technical correctness of every derivation and quoted result.** Especially WB2 §4.3 (Hastie et al. Theorem 1 as quoted), §5.1–5.4 (sandwich, TIC factor of two, Fisher identity, BIC, Watanabe's regular-case Bayes/Gibbs constants), §6.3 (Lyapunov solution and the "matches a tempered posterior iff $C\propto I$" conclusion; check the scaling conventions for $\eta/B$ and $n$), WB3 §5.1 (Chinchilla allocation), WB4 §2–3 (invariance theorem, uncomputability, Kraft, dominance, the regret bound, Legg's result as paraphrased), WB5 §3.2 (Roeder–Metz–Kingma statement), §5.1 (JL count), §6.1 (spectral scaling exponent). Flag anything stated more generally than its theorem.
2. **Difficulty calibration.** The cohort is strong; the instructor wants problems harder than the Physics day. Say which problems are too easy, too hard, or too long, and which have no clear "done" state.
3. **Prerequisite chains.** Each section after WB2 §1 should be doable standalone. Flag hidden dependencies.
4. **Symmetry of presentation.** Where the spec supplies a verdict before the student investigates, say so and propose the symmetric version.
5. **Omissions for practitioners.** What a working alignment researcher needs from "theory of deep learning" that none of WB1–WB6 covers. (The earlier Codex critique nominated evaluation under selection and distribution shift, and reward-proxy optimization.)
6. **Build feasibility.** Are the supplied artefacts (WB2 §7 sweep, §6.2 scatter, §4.3 figure, WB4 §4.1 count, WB5 §5.2 phase diagram) well specified enough to generate? Anything that needs more than a few GPU-hours on Modal?
7. **Readings.** Dead or wrong links; readings that duplicate; readings missing for a section; comprehension questions that cannot be answered from the reading.

## Known open items
- WB4: the instructor is not a specialist in Solomonoff induction. The teaching note from the link-gathering pass is reproduced at the end of the spec file; check it.
- WB5 §3.2: the instructor mentioned a "perfect PRH paper that is good to work through"; the spec guesses this is the linear-identifiability line (Roeder–Metz–Kingma; Reizinger et al.). Confirm with the instructor.
- Code policy: markdown-only with supplied helper cells, as in the Physics day, or allow optional coding extensions. Instructor's call.
- WB2 §7 sweep and §6.2 scatter need to be generated (Modal). Specs are in the build notes.
- Cohort size (affects the 17:00 convergence format) and which day of the week.

## What to deliver back
A numbered list of findings, each with a one-line recommended change, grouped by the seven headings above. Under 2000 words. Do not rewrite the spec; the instructor will revise.

## Appendix: Solomonoff teaching note (from the link-gathering pass; check it)

Start with Bayesian Occam (MacKay ch. 28): a model's evidence trades fit against the volume of parameter space it must commit to, so simplicity is a theorem of Bayes, not an extra axiom. Then make simplicity objective: Kolmogorov complexity $K(x)$ is the length of the shortest program printing $x$ on a fixed universal machine, defined only up to an additive machine-dependent constant (invariance theorem) and uncomputable. Solomonoff's prior $M(x)=\sum_{p:U(p)\text{ starts with }x}2^{-|p|}$ turns "prefer short programs" into a prior over infinite sequences that dominates every computable predictor and converges with total expected log-loss regret bounded by about $K(\mu)\ln2$ bits for true environment $\mu$. $M$ is a lower-semicomputable semimeasure, not a computable measure, so every practical use is a computable surrogate (speed prior, MDL, AIXItl), and Legg (2006) shows this is not a technicality: predictors that learn all sequences up to a given complexity must be comparably complex, and past a moderate level you cannot prove they work. The claim "NNs ≈ Solomonoff" is best stated as a claim about the parameter-function map: random initialization induces a prior $P(f)$ over functions on a finite domain, and Valle-Pérez / Dingle / Mingard show $P(f)\lesssim2^{-a\tilde K(f)+b}$ for a computable proxy $\tilde K$ (LZ, entropy); combined with the empirical, regime-limited observation that SGD approximately samples this prior's posterior, you get an Occam-like learner with no explicit program search.

Strongest links: the simplicity bias of the prior itself (robust across architectures, with an algorithmic-information basis in Dingle et al.) and the compression/prediction equivalence (Delétang et al.). Weakest links: the bound is one-directional (high probability implies simple, but most simple functions get negligible mass, so "biased toward simple" is not "Solomonoff-like weighting of all simple hypotheses"); the proxy is LZ or Boolean entropy, not $K$, and the domain is functions on a finite set, not sequence-generating programs; "SGD ≈ Bayesian sampler" fails in the feature-learning and large-step regimes that matter (see WB2 §6); Grau-Moya et al. is the only direct test and is small-scale.

Common errors: $K$ is only defined up to $O(1)$ and is uncomputable, so "the $K$-complexity of this network" is never a number; $M$ is a prior over sequences, the NN prior is over functions, and they are related only by fixing an input enumeration; $M$ is a semimeasure (mass leaks to non-halting programs); Bayesian Occam (volume) and Kolmogorov Occam (description length) are different mechanisms that happen to agree, and MDL's two-part code is a third; Solomonoff's optimality is relative to computable environments and does not evade Legg's result or NFL, which is why Goldblum et al. ("real data is low-complexity, so NFL is not binding") is the right framing.
