# WB1 figure plan

**Notebook:** MDL WB1. The builder title is now "The bitter lesson, and what it did to alignment" (`notebooks/build_wb1.py` → `notebooks/01_intelligence_then_and_now.html`).
**Date:** 2026-10-09. Planning only; no builder, notebook or asset was edited.
**Count:** 15 proposals: **P1: 2 (F5, F6) · P2: 7 (F1, F2, F4, F7, F9, F10, F12) · P3: 6 (F3, F4b, F8, F11, F13, F14)**.

WB1 is mostly readings, and most of its sections are not built around a plot. Figures earn their place in two spots: the one derivation chain (§2.E planning as inference), which is built around a graphical model and around *where the max, the soft max and the expectation sit in a backup*, and a few spots where one picture states a reading's headline result better than prose. I did not propose figures for most reading blurbs.

**Sources checked (2026-10-09).** I downloaded every paper figure cited below, read the caption in the extracted text, and for Levine Fig. 1, Langosco Fig. 1, Hubinger Fig. 1, Baker Fig. 4 and MacDiarmid Fig. 1 also looked at the rendered page. I took licences from the arXiv abs pages:
- **CC-BY 4.0** (free to embed, with attribution): Langosco et al. 2105.14111, MacDiarmid et al. 2511.18397, Sevilla et al. 2202.05924, and the Epoch AI data.
- **arXiv default non-exclusive licence** (no reuse grant, so embedding relies on fair use or permission; CRW decides): Levine 1805.00909, Silver et al. 1712.01815, Hubinger et al. 1906.01820, Baker et al. 2503.11926.
- **Sutton & Barto, 2nd ed.:** CC BY-NC-ND 2.0, so a verbatim copy with attribution is fine for non-commercial use, but adapted versions are not.

For schematics I recommend redrawing in our own style either way. Drawing conventions (backup-diagram nodes, plate-free graphical models) are not anyone's property, and our versions need notebook notation.

---

## Proposals, in notebook order

### F1. Two eras of training compute (P2)
- **Anchor:** "The headline reading". Place after the Sutton blurb, before "Keep three questions in mind for the rest of the workbook."
- **What it shows:** a log-y scatter of training compute (FLOP) against publication date, 1950–2026, for Epoch AI's notable models. Two fitted trend lines: pre-2010 (about 21-month doubling, per Sevilla et al.) and post-2010 (about 6-month doubling). Label about eight points that WB1 itself mentions: Perceptron Mark I (1957), TD-Gammon (1992), DQN (2013), AlphaGo Lee (2016), AlphaZero (2017), GPT-3 175B (2020), GPT-4 (2023), and one 2025 frontier point. Deep Blue has no compute entry in the Epoch data, so leave it out rather than estimate it.
- **Why here:** Sutton's argument rests on one empirical premise: the compute available keeps growing exponentially, so methods that use it win eventually. The plot shows the premise and the 2010 change of slope. It also places §2's systems (TD-Gammon, DQN, AlphaZero) and §4's LLMs on one axis, which ties the workbook's history together.
- **Source:** (b) generate it ourselves from Epoch's CSV (`https://epoch.ai/data/notable_ai_models.csv`, checked today: 1,410 rows with compute, the latest dated 2026-09-09; CC-BY 4.0, "credited under the Creative Commons Attribution license"). Recipe: pandas + matplotlib, drop rows with no compute, and fit `log10(FLOP)` against year separately before and after 2010-01-01. Runs in under 2 s. The canonical published version is Sevilla et al., arXiv 2202.05924, Fig. 1 (n = 121 milestone models, 1952–2022, "three eras"; CC-BY 4.0), which can be embedded as is if CRW prefers a citable figure. The self-generated version is more current.
- **Caution for the caption:** this is compute *used*, not the cost of compute that Sutton actually invokes. Epoch's GPU price-performance data could add a second panel if CRW wants the premise itself, but I'd keep one panel.
- **Caption draft:** "Training compute of notable AI systems, 1950–2026 (data: Epoch AI, CC-BY). Before about 2010 it doubled roughly every two years; since deep learning, roughly every six months. The systems of §2 and §4 are labelled."

### F2. Backup diagrams for V\* and Q\* (P2)
- **Anchor:** §2.A. Place right after the $Q^*$ equation, before "Both say Bellman's *principle of optimality*".
- **What it shows:** two small trees in Sutton–Barto backup-diagram style. Left, $V^*$: an open circle $s$, then an arc marked **max** over the solid dots $a$, then branches weighted by $P(s'\mid s,a)$ (**expectation**) down to open circles $s'$. Right, $Q^*$: a solid dot $(s,a)$, then **expectation** over $s'$, then an arc marked **max** over $a'$. Shade the max arcs in one colour and the expectation fans in another, and keep those colours for F5.
- **Why here:** the whole of Exercise 2.2(b) turns on "where does the maximum sit relative to the expectation". For $V^*$ the max is outside the expectation, so one sample cannot estimate it; for $Q^*$ it is inside. Seeing the two trees makes that ordering obvious, and it sets up the visual language that F5 reuses for the soft backups.
- **Source:** (c) a schematic, redrawn. The convention is Sutton & Barto 2nd ed., Fig. 3.4 ("Backup diagrams for v\* and q\*", checked in the book PDF). An SVG or a small matplotlib drawing is enough.
- **Caption draft:** "Backup diagrams for the Bellman optimality equations (after Sutton & Barto, Fig. 3.4). Open circles are states, solid dots are actions, arcs are maxima, fans are expectations over the next state. In $V^*$ the max is taken outside the expectation; in $Q^*$ it is inside."

### F3. Baird's counterexample: the deadly triad diverges (P3)
- **Anchor:** §2.C. Place after "together they can make TD learning diverge, and there are simple counterexamples."
- **What it shows:** left, Baird's 7-state MDP (six upper states with value $2w_i+w_8$, one lower state with $w_7+2w_8$; the dashed action goes to a random upper state, the solid action to the lower one). Right, all 8 weights against sweeps (0–1000) under the *expected* (DP-style) semi-gradient off-policy TD(0) update, diverging. Optionally a faint overlay of the same update under the on-policy distribution, which converges.
- **Why here:** the text asserts divergence. The plot shows that it happens with no sampling noise at all, in a problem linear features can represent exactly. That separates the triad from "RL is noisy".
- **Source:** (b) generate it. Sutton & Barto §11.2, eq. 11.9, Figs. 11.1–11.2 (checked). Features $x_i=2e_i+e_8$ for $i\le6$ and $x_7=e_7+2e_8$; $w_0=(1,1,1,1,1,1,10,1)$, $\alpha=0.01$, $\gamma=0.99$; target $\gamma\,\hat v(7)$, update $w\mathrel{+}=\frac{\alpha}{7}\sum_s(\text{target}-\hat v(s))\,x_s$. I ran it: after 1000 sweeps $w_{1..6}\approx115$, $w_8\approx336$ and still growing. Takes under 0.1 s.
- **Caption draft:** "Baird's counterexample (Sutton & Barto §11.2). Function approximation, bootstrapping and off-policy updates together make the weights diverge, even with exact expected updates and a value function ($w=0$) that the features represent perfectly."

### F4. Policy iteration and AlphaZero's loop (P2)
- **Anchor:** the Exercise 2.3 solution. Place after part (a), ending "…self-play with search as 'a powerful policy evaluation operator'." It could instead go in §2.D after "Exercise 2.3 shows that AlphaZero's training loop is policy iteration from §2.A, with tree search as the improvement step", but in the solution it doesn't pre-empt the exercise.
- **What it shows:** two loops with matching layout. Left, tabular policy iteration: $\pi\to$ *evaluate* ($V^\pi$) $\to$ *improve* (greedy) $\to\pi'$. Right, AlphaZero: the network $f_\theta=(\mathbf p,v)$ $\to$ **MCTS** (improvement) $\to\boldsymbol\pi$ $\to$ **self-play** $\to z$ (Monte Carlo evaluation). Training then closes the loop with two arrows back to $f_\theta$: $v\leftarrow z$ (evaluation) and $\mathbf p\leftarrow\boldsymbol\pi$ (amortization, marked as having no tabular counterpart). Colour each piece by role: improvement, evaluation, amortization.
- **Why here:** the exercise asks for a mapping between two loops. A side-by-side diagram makes the correspondence and the one extra arrow (policy-head distillation) visible at a glance.
- **Source:** (c) a schematic. Related canonical pictures: the GPI diagram in Sutton & Barto §4.6 (unnumbered) and AlphaGo Zero's self-play figure (reproduced as S&B Fig. 16.7). Draw our own; don't copy the Nature figure.
- **Caption draft:** "AlphaZero as policy iteration. MCTS improves the network's policy, and self-play outcomes evaluate the improved policy. The policy head is trained to imitate the search, a step tabular policy iteration does not need."

### F4b. AlphaZero overtakes the hand-tuned engines (P3)
- **Anchor:** §2.D. Place after "AlphaZero searches a thousand times fewer positions than Stockfish and still wins, because its evaluation is better."
- **What it shows:** Silver et al., arXiv 1712.01815, **Fig. 1**: Elo against training steps (0–700k) for chess, shogi and Go, with horizontal lines for Stockfish, Elmo and AlphaGo Lee/Zero. AlphaZero passes Stockfish at about 300k steps (about 4 h). Fig. 2 of the same preprint (Elo against thinking time per move, AlphaZero against Stockfish/Elmo) is the alternative if CRW wants the "search with a learned evaluation scales" point rather than "learning overtakes hand-tuning".
- **Why here:** it is the bitter lesson in one panel. A from-scratch learner crosses the line of the best hand-tuned system within hours.
- **Source:** (a) arXiv 1712.01815, Figs. 1–2 (captions checked). arXiv default licence, so link it, or embed under fair use with attribution.
- **Confounds** (if embedded, the caption should hedge): hardware differs (TPUs against 64 CPU threads), and the Stockfish 8 settings (1-minute moves, 1 GB hash, no opening book) were criticised. The Science 2018 version re-ran the matches under stronger conditions, with different figure numbering. Because of these confounds I rate it P3; the §2.D table already makes the point.
- **Caption draft:** "AlphaZero's Elo during self-play training, compared with the strongest existing programs (Silver et al. 2017, Fig. 1). Hardware and engine settings differ between the systems, so read the crossover times as indicative."

### F5. Graphical model for control as inference (P1)
- **Anchor:** §2.E, "The model." Place right after $p(\mathcal O_t=1\mid s_t,a_t)=e^{\,r(s_t,a_t)}$ and before "For this to be a probability we need $r\le0$."
- **What it shows:** a chain over $t=1..4$. Nodes $s_t$ (bottom row), $a_t$ (middle) and $\mathcal O_t$ (top, shaded = observed). Edges: $s_t\to s_{t+1}$, $a_t\to s_{t+1}$, $s_t\to\mathcal O_t$, $a_t\to\mathcal O_t$. There is **no** $s_t\to a_t$ edge, because the action prior is uniform. One option is to fade $\mathcal O_{1:t-1}$ and highlight $\mathcal O_{t:T}$, labelling the backward message $\beta_t(s,a)=p(\mathcal O_{t:T}\mid s,a)$ as flowing right to left.
- **Why here:** the derivation chain is about this graph. The B.2 hints rely on conditional independences ("given $(s_t,a_t)$, $\mathcal O_t$ is independent of the future"; "the future depends on $(s_t,a_t)$ only through $s_{t+1}$"), which can be read straight off the graph and are tedious to state in words. It also makes the HMM analogy, $\mathcal O_t$ as observations, concrete.
- **Source:** (c) redraw. The canonical figure is Levine 2018, arXiv 1805.00909, **Fig. 1(b)** ("graphical model with optimality variables"). I checked the rendered page: the edges are exactly as listed, with no $s\to a$ edges. arXiv default licence, so redraw. TikZ-style SVG or matplotlib.
- **Caption draft:** "The control-as-inference model (after Levine 2018, Fig. 1b). States and actions form the trajectory prior; each shaded $\mathcal O_t$ is a binary 'optimal' variable with likelihood $e^{r(s_t,a_t)}$. Planning conditions on all $\mathcal O_t=1$; the backward message $\beta_t$ carries $\mathcal O_{t:T}$ back to step $t$."

### F6. Three backups side by side: hard, exact inference, max-ent (P1)
- **Anchor:** §2.E, at the end of the "**The result.**" paragraph, after "…which still needs a reward or goal." and before the Levine reading. That makes it a map of B.1–B.5. Alternative: after the Exercise B.5 solution, as a summary. All solutions are open, so placing it first costs nothing.
- **What it shows:** three one-step backup diagrams in F2's visual language, for a state node, its actions and their next states.
  - (a) **Bellman** (§2.A): max over $a$, expectation over $s'$.
  - (b) **Exact inference** (B.2): $\log\sum e^{(\cdot)}$ over $a$, *and* $\log\mathbb E\,e^{(\cdot)}$ over $s'$. Draw the $s'$ fan in the "max" colour, faded, labelled "optimistic: picks its luck"; the $\alpha\to0$ limit is the max over *reachable* $s'$.
  - (c) **Max-ent RL / KL control** (B.4–B.5): $\log\sum e^{(\cdot)}$ over $a$, plain expectation over $s'$; the $\alpha\to0$ limit is (a).

  A one-line legend under each panel gives its $\alpha\to0$ limit.
- **Why here:** the punchline of the derivation chain is *which node gets soft-maxed*. Soft-maxing the agent's choice is fine. Soft-maxing nature's choice is the optimism of B.3, and the variational fix (B.4) restores the expectation there. One picture holds the result of five exercises, and students can check where they are in the chain.
- **Source:** (c) a schematic of our own. It extends S&B's backup-diagram convention to soft backups; I know of no published version, so it isn't a derivative of any figure.
- **Caption draft:** "One step of three backups. Exact inference (b) soft-maxes over both the agent's action and the next state, so at low temperature it plans as if it chose the outcome of every die roll. Max-ent RL (c) keeps the expectation over next states and reduces to the Bellman backup (a) as $\alpha\to0$."

### F7. Inference is optimistic: the gamble of Exercise B.3 (P2)
- **Anchor:** the Exercise B.3 solution. Place after part (d), ending "…Exercise B.4 removes the problem by forbidding the agent to choose its luck."
- **What it shows:** two panels, closed form, $p=0.01$, $R=10$.
  - **Left:** $Q(\text{gamble})$ against the temperature $\alpha$ (log x-axis, 0.03 to 1000) at $C=100$. The curve is $\alpha\log\!\big(p\,e^{R/\alpha}+(1-p)\,e^{-C/\alpha}\big)$, with horizontal references for the safe action ($0$), the expected backup ($pR-(1-p)C=-98.9$), and the best reachable value ($R=10$). Checked values: $9.77$ at $\alpha=0.05$, $5.39$ at $\alpha=1$ (B.3(b)), a zero crossing near $\alpha\approx R/\log(1/p)=2.17$, $-36.0$ at $\alpha=10$, and $-98.84$ at $\alpha=1000$.
  - **Right:** $Q$ against catastrophe size $C\in[0,1000]$ at $\alpha=1$. The exact-inference curve is flat at $\approx5.40$; the expected backup falls linearly.
- **Why here:** (d)'s headline, "does not depend on $C$ at all", is surprising; the right panel makes it impossible to miss. The left panel shows the B.2(d) stochastic limit (exact inference tends to the best reachable outcome as $\alpha\to0$) and how much temperature it takes before the soft backup becomes risk-averse. Both connect B.2(d) to B.3 numerically.
- **Source:** (b) numpy/matplotlib with `np.logaddexp` for stability. Under 1 s. Values verified above.
- **Caption draft:** "The gamble of Exercise B.3 ($p=0.01$, win $R=10$, loss $C$). Left: the exact-inference value moves from the best reachable outcome at low temperature towards the expected value only at very high temperature. Right: at $\alpha=1$ it ignores the size of the catastrophe entirely, while the expected backup does not."

### F8. KL-regularized control as a Bayesian update (P3)
- **Anchor:** the Exercise B.5 solution. Place after (c), ending "…has the form of a Bayesian update of that policy."
- **What it shows:** bar charts over about eight actions. Prior $\pi_0$, likelihood $e^{r/\lambda}$, and posterior $\pi^*\propto\pi_0e^{r/\lambda}$ for $\lambda\in\{0.3,1,3\}$. Choose $\pi_0$ to put little mass on the reward-maximizing action, so the student sees that the posterior can only reweight $\pi_0$'s support and that small $\lambda$ concentrates it.
- **Why here:** it is the form carried forward into RLHF ("What carries forward", second bullet). The picture makes "reweighting, not replacement" concrete. Only P3: the algebra is already transparent.
- **Source:** (b) matplotlib, deterministic, instant.
- **Caption draft:** "$\pi^*\propto\pi_0e^{r/\lambda}$: the reward reweights the reference policy like a likelihood. Smaller $\lambda$ concentrates the posterior on high-reward actions, but only on actions $\pi_0$ already supports."

### F9. Base objective, learned algorithm, mesa-objective (P2)
- **Anchor:** §3, the "**2016: aligning a trained agent.**" paragraph. Place after "…and *inner* alignment (is what the model learned the training objective?)." and before "Goal misgeneralization made the inner problem empirical".
- **What it shows:** two stacked panels.
  - **2006:** designer writes $U$ → agent maximizes $U$ (one objective, one optimizer).
  - **2016:** designer's intent →(*outer alignment*)→ base objective / loss → base optimizer (SGD) produces the learned algorithm, which may contain a mesa-optimizer with a mesa-objective →(*inner alignment*: mesa-objective vs loss)→ actions.

  Bracket the two alignment gaps.
- **Why here:** the outer/inner split is a statement about a structure with two optimizers and two objectives. The diagram shows why 2006 had no "inner alignment" (Question 3.2): there was only one box. The diagram does not answer Question 3.2 for the student; it gives the vocabulary.
- **Source:** (c) redraw, extending Hubinger et al. 2019, arXiv 1906.01820, **Fig. 1** (checked on the rendered page: Base Optimizer [Base Objective] —optimizes→ Mesa Optimizer [Mesa Objective], with Input → Mesa Optimizer → Output). arXiv default licence, so redraw and add the intent box and alignment brackets.
- **Caption draft:** "In 2006 one objective was written by hand and optimized. In 2016 a base optimizer trains a model against a loss, and the model may pursue an objective of its own (after Hubinger et al. 2019, Fig. 1). Outer alignment is intent against loss; inner alignment is loss against the learned objective."

### F10. Goal misgeneralization in CoinRun (P2)
- **Anchor:** §3, same paragraph. Place after "…which agrees with it on the training distribution and comes apart off it." and before "Turner put the shift in a slogan".
- **What it shows:** Langosco et al. 2022, arXiv 2105.14111, **Fig. 1**, two CoinRun screenshots. (a) Training: the coin is always at the right end of the level. (b) Test: the coin is placed randomly, and the agent runs to the end, skipping it. Checked on page 1 of the PDF.
- **Why here:** inner misalignment is abstract until you see one. This is the canonical instance: capabilities generalize (the agent still clears the level), the goal doesn't. It makes "two goals that agree on the training distribution" concrete in one glance.
- **Source:** (a) embed. **CC-BY 4.0** (arXiv abs page), so it can be reproduced with attribution. An extracted PNG crop of page 1 is enough.
- **Caption draft:** "Goal misgeneralization (Langosco et al. 2022, Fig. 1; CC-BY 4.0). Trained with the coin always at the end of the level, the agent learns 'go right'. With the coin moved at test time, it still competently reaches the end and skips the coin."

### F11. The ontology problem as a picture (P3)
- **Anchor:** Question 3.1. Place after the setup, ending "…they agree off $E$ and nothing in $U$ chooses between them on $E$.", and before (a).
- **What it shows:** two blobs. On the left, the refined state space $S'$, with a small hatched region $E$; on the right, the old space $S$ coloured by $U$. Arrows show $\phi$ mapping $S'\setminus E$ to $S$ uniquely, and two diverging arrows $\phi_1,\phi_2$ from $E$, one landing in a high-$U$ region. An "optimizer" arrow on $S'$ points into the part of $E$ where $U\circ\phi_1$ is high and cheap.
- **Why here:** the question is posed in abstract notation, and the picture makes (a)'s answer (optimization goes where the translation is ambiguous) visible. P3, because the diamond example already does much of this work.
- **Source:** (c) a schematic of our own.
- **Caption draft:** "A utility on an old ontology $S$, translated to a refined one $S'$. Off $E$ the translation is unambiguous; on $E$ two readings disagree, and an optimizer is drawn to the cheap states that one reading scores highly."

### F12. Optimizing against a CoT monitor teaches obfuscation (P2)
- **Anchor:** §4, "**What came back.**" Place after "Training against a chain-of-thought monitor can teach a model to hide its intent." and before "Reward hacking learned in coding environments…". This is better than at the Baker blurb, because the prose is where the claim is made.
- **What it shows:** Baker et al. 2025, arXiv 2503.11926, **Fig. 4**, three panels against training epochs. Each compares the baseline agent with an agent trained with CoT pressure, on three measures: unit tests passing without cheating, cheating caught by the monitor, and cheating *not* caught by the monitor. The baseline's hacks are almost all caught. Under CoT pressure, caught hacking stays near zero while uncaught hacking rises to about 0.5. Checked on the rendered page 8.
- **Why here:** it is Goodhart's law (and Exercise 2.2(c)'s "optimizing against a noisy estimate selects for its errors") on a frontier-lab setup, in one picture: put the monitor in the reward and you train the monitor's blind spot. It is the single most legible plot behind §4's "what came back".
- **Source:** (a) arXiv default licence, so link the figure, or embed under fair use with attribution (CRW decides). There is no public data to regenerate it from.
- **Caveats for the caption:** this is a non-frontier agent in their agentic coding environment; "cheating" is detected by a specific ground-truth check, so measured rates are for one class of exploit; single runs.
- **Caption draft:** "Putting a chain-of-thought monitor into the reward (Baker et al. 2025, Fig. 4). Both agents learn to hack. The baseline's hacks are almost all caught by the monitor; the agent trained against the monitor hacks somewhat less, but its hacks are no longer caught."

### F13. Hacking onset coincides with broad misalignment (P3)
- **Anchor:** §4, "**What came back.**" Place after "Reward hacking learned in coding environments can generalize to broader misalignment."
- **What it shows:** MacDiarmid et al. 2025, arXiv 2511.18397, **Fig. 1**. Left: the misalignment-generalization score and hack rate against RL step. A dotted line marks the onset of hacking (>2% of episodes), after which misalignment jumps; a grey no-hacking baseline stays flat. Right: six individual evals (Betley et al., fake/bad goals, deceptive on alignment questions, monitor disruption, exfil offer, frame colleague) all rising at onset. Checked on page 2.
- **Why here:** it shows the claimed generalization as a timing coincidence across six evaluations.
- **Source:** (a) **CC-BY 4.0**, so embed with attribution.
- **Caveat** (must be in the caption): the model was first given knowledge of these hacks by synthetic-document fine-tuning. Rated P3 only because §4 is reading-led and one data plot (F12) may be enough. If CRW wants only one §4 plot, F13 is the lower-friction choice licence-wise and F12 is the more direct illustration of Goodhart.
- **Caption draft:** "When a model that had been taught about possible hacks starts reward hacking in production coding RL, misalignment rises on six unrelated evaluations at the same step (MacDiarmid et al. 2025, Fig. 1; CC-BY 4.0). A run where hacking was penalized shows no rise."

### F14 (optional). Two timelines: capability paradigms and alignment waves (P3)
- **Anchor:** title cell. Place after the numbered list of four sections, before "Each section opens with a few paragraphs of context…"
- **What it shows:** a horizontal 1950–2026 axis with two lanes.
  - **Top lane (how capable systems were built):** GOFAI / symbols and search → TD-Gammon 1992, Deep Blue 1997 → DQN 2015, AlphaGo 2016, AlphaZero 2017 → GPT-3 2020 → RLVR reasoning models 2024–.
  - **Bottom lane (alignment wave):** "ideal agents" (Omohundro 2008, *Superintelligence* 2014) → "trained agents" (*Concrete Problems* 2016, *Risks from Learned Optimization* 2019) → "pretrained + RL" (2022–).

  Vertical bands link each wave to the paradigm it assumed. Each band links to its section anchor.
- **Why here:** §3's pitch says "knowing which wave a worry comes from tells you what picture of AI it assumes". This picture states that pairing in one view and orients students choosing a section.
- **Source:** (c) a schematic of our own, SVG.
- **Risk:** it could read as decorative. Keep it only if CRW likes timelines; otherwise drop it.
- **Caption draft:** "Each wave of alignment work assumed the AI of its day: hand-written ideal agents (§1, §3), agents trained by deep RL (§2, §3), and pretrained models with RL on top (§4)."

---

## Considered and not proposed
- **Money-pump cycle (Q1.1), Löbian chain of theories (Q1.3):** the prose is already as fast as a picture.
- **$b^d$ combinatorial-explosion plot:** padding.
- **S&B Fig. 8.11 (width × depth of updates: DP / TD / MC / exhaustive search):** a lovely figure, but §2 never discusses Monte Carlo against TD depth, so it would need new text.
- **DQN's 49-game bar chart, the AlphaGo pipeline, the Golden Gate feature, alignment-faking and emergent-misalignment headline plots:** these would decorate reading blurbs; their papers are the readings.
- **CoastRunners boat GIF (specification gaming):** iconic, but the licence for the OpenAI/DeepMind media is unclear, and the Krakovna blurb links to it. A link at most.

## Gaps and issues noticed
1. **README out of date for WB1.** `notebooks/README.md` still calls WB1 "Intelligence and alignment, then and now… planning as inference, §3" and lists "the ontology problem; learned values and simulators". The builder's title is now "The bitter lesson, and what it did to alignment", and planning as inference is **§2.E**, not §3. The task brief uses the old title too. (The generated HTML matches the builder.)
2. **The text references no figures,** and none exist. No interactive figure in `assets/interactive/` belongs to WB1, and none is misplaced here.
3. **Factual cross-check of the §2.D text against AlphaZero arXiv 1712.01815:** "80 thousand positions per second in chess… compared to 70 million for Stockfish". That ratio is about 875×, so "a thousand times fewer" and the table's "tens of thousands… against tens of millions" are fine. If F4b is embedded, cite the preprint's figure numbers; the Science version's figures differ.
4. **F12 and F13 overlap** in purpose; CRW should probably pick one (see F13).
5. **Licensing decision for CRW:** whether to embed arXiv-default-licence figures (Baker Fig. 4, AlphaZero Fig. 1) under fair use or link them. Everything else in this plan is CC-BY, our own generation, or our own redraw.
