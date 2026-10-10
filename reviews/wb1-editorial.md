# WB1 editorial round (2026-10-08)

Owner: Claude Opus 5.5, acting as WB1 owner. Only `notebooks/build_wb1.py` and this file were edited. No notebooks were regenerated and no `.ipynb` was touched. The previous owner's corrected §3 math was kept. The pre-rewrite snapshot (`/tmp/mdl-before-20261008/notebooks/build_wb1.py`) was not used.

## 1. Initial critique (current builder vs `reading-format-check.md` and `technical-sources.md`)

**Already done before this round**
- technical-sources "First rewritten-source recheck" item 1: the 3.4(b) prompt already said "expectation minus $T\log|A|$".
- Item 2: the 3.4(c) answer already used "whenever $\beta_{t+1}$ is not constant on the reachable next states" and "generally lies outside the family", and the policy was time-indexed $\pi_t$.
- Item 3: the 3.5(c) answer already used the $b\ge\max r$ normalization with evidence $e^{-b}Z$.
- "Added audit" points: the stochastic zero-temperature limit (best reachable state), $T\log|A|$ in the KL and ELBO, the fixed start, finite $A$, and the one-step vs sequential scope were all present.
- The §3 opening already separated control from generative-model fitting.

**Remaining**
- The intro still promised a "which survive" synthesis (§6), and its route listed §6.
- §1 was about 400 words of narration.
- §2 was about 300 words of narration (§2.A–B and "The result").
- §3.C was about 180 words.
- §4 was about 700 words: §4.A on VNM, Goodhart, instrumental convergence and deception; the diamond narrative; and §4.C. The Wentworth blurb pointed at "the three claims above".
- §5 opened with "The result" stating the contested "relocates" thesis.
- Q5.1 was a case-for/case-against/"what would decide it" rubric.
- §6 held a synthesis table and pro/con essay.
- The Q2.1 answer still had the falsification-boilerplate bullet.
- The closers pointed at §4.C, Q5.1, §5.D and the §6 table.
- "What carries forward" wrote $e^{r/\beta}$, while Exercise 3.5 uses $\lambda$.
- Exercises 3.2(d) and 3.5(b) cited "§1's Bellman equation".
- The 3.2(d) answer used $Q^*_t$ without defining it.

## 2. Edits

**Intro**
- New route (§1–5). The intro claims only a common inference formulation of planning and max-ent RL under stated assumptions.

**§1**
- One sentence plus a 3-row table: picture | formal object | written by hand. Q1.1 and its answer are kept.

**§2**
- Two sentences, then the Sutton blurb.
- The falsification bullet in the Q2.1 answer is deleted.

**§3**
- The Levine blurb is now at the top, and "The result" is tightened. It includes the fixed start state and states that generative-model fitting is not the same objective as control.
- The hard finite-horizon Bellman optimality equation (with $V^*_{T+1}\equiv0$) is now a displayed line in §3.A and contrasted with the infinite-horizon discounted form. Exercises 3.2(d) and 3.5(b) now cite §3.A.
- 3.2(d) answer:
  - defines $Q^*_t$ and the base case;
  - the stochastic case now states the limiting recursion $V^+_t$;
  - adds the bound $-(T-t)\alpha\log(1/p_{\min})\le\tilde V_t-V^+_t\le(T-t+1)\alpha\log|A|$.
- 3.2 hint (b): $\mathcal O_{1:t-1}\perp(a_t,\mathcal O_{t:T})\mid s_t$.
- 3.3 setup wording: "shifts both backups in (a) by that constant".
- 3.4 hint displays the normalized posterior, which the exercises notebook otherwise hides in a blank answer.
- 3.5 title is now "as a Bayesian update".
- 3.5(b) prompt asks for the zero-temperature limit.
- 3.5 answer adds the sequential KL-control solution $V_t=\log\sum_a\pi_0e^{Q_t}$, $\pi_t=\pi_0e^{Q_t-V_t}$.
- 3.1(b) answer: feasibility includes the start state.
- §3.C is cut to two sentences, which say the derivation does not settle which route scales.

**§4**
- Pitch sentence, then the rewritten Wentworth blurb (naming abstractability, human-compatibility and convergence), then "Setup for Question 4.1", then Q4.1.
- §4.A, the diamond narrative and §4.C are deleted.
- In the sketch:
  - $\pi\to\phi$ and $A\to E$, to avoid clashing with §3's policy and action set;
  - "may disagree";
  - the argmax claim is replaced (see finding 2 below).

**§5**
- Neutral one-line pitch pointing to WB3 §6 for simulators (janus; Nardo), then the three blurbs.
- §5.D is removed. WB3 now cites only "WB1 §5" for this.
- Q5.1 is deleted.
- Byrnes and Retarget are now *optional*; Hubinger stays *core*.
- The Byrnes and Hubinger one-line cross-links are folded into their blurbs' "about" text.

**§6 and closers**
- §6, its route line and the `wb1-section-6` anchor are deleted. Nothing in the repo's builders links to it.
- "What carries forward" is re-pointed and uses $\lambda$.
- The Hubinger §4 bullet, which duplicated the blurb, is removed from Further study.

**Source check by owner.** I read Levine's arXiv PDF (pp. 1–16). Eqs. (4), (6), (7), (9), (11), (15)–(16) and (19), §2.3–2.5, §3.1–3.2, §4 and §5 (Kalman duality, LMDPs, KL-constrained policy search, intent inference) match the builder's citations.

## 3. Independent critique (fresh Opus sub-agent, read-only, unbiased prompt)

It found no blockers. It re-derived all §3 algebra and numbers and found them correct. It also reported checking Sutton, Byrnes post 9, Hubinger §1/§2.1/§4/§7 and the Retarget post against the sources. Findings and dispositions:

| # | Finding | Disposition |
|---|---|---|
| 1 | Every reading was marked *core*, so the label was meaningless | Fixed: §5 now has Hubinger core, Byrnes and Retarget optional. Single-reading sections keep core. |
| 2 | §4 argmax claim wrong: $U\circ\pi_i$ only takes values of $U$, so it is not "more extreme" on $A$ | Fixed: replaced with a reachability/cost argument; "may disagree"; renamed symbols; Q4.1 answer adjusted. |
| 3 | §3 result omitted the fixed start state | Fixed. |
| 4 | §4 blurb not at top; "two-line" was inaccurate | Fixed: blurb under the pitch; sketch labelled "Setup for Question 4.1". |
| 5 | §3 blurb after about 210 words; scope stated three times | Fixed: blurb first; result tightened; intro reduced to one scope clause. |
| 6 | 3.4 hint pointed to a formula visible only in a solution | Fixed: formula displayed in the hint. |
| 7 | Sequential KL-control scope sentence used the uniform-prior backup | Fixed: $\pi_0$-weighted $V_t$ and $\pi_t$. |
| 8 | Breadth: no source for the ontology problem or decision theory; Korbak et al. for 3.5 | **Not adopted; deferred to CRW.** Only papers CRW knows should be added. Candidates: de Blanc 2011 (arXiv 1105.3821), Omohundro 2008, Korbak–Perez–Buckley 2022 (the critic did not verify that arXiv ID). |
| 9 | 3.5(b) did not ask for the zero-temperature limit | Fixed. |
| 10 | Bellman wording mixed finite and discounted cases | Fixed. |
| 11 | Stochastic limit was loose ($\tilde V$ depends on $\alpha$) | Fixed: limiting recursion $V^+$ plus two-sided bound. |
| 12 | 3.2(b) hint conditional-independence statement incomplete | Fixed. |
| 13 | "shifts both backups below" was ambiguous | Fixed. |
| 14 | §3.C "fits the trajectory distribution" was vague | Fixed: "including a prior over behaviour like the $\pi_0$ of Exercise 3.5". |
| 15 | Byrnes blurb long; post-9 line duplicated it | Fixed: merged; §9.4/§9.5 parentheticals dropped. |
| 16 | WB3 no longer cites §5.D; the stub was redundant | Fixed: §5.D removed; pointer folded into the §5 lead. |
| 17 | §3 title poses a question the section does not settle | Route line now says so. Title kept: it is a question, not a claim. |
| 18 | Further-study duplicate; "is Bayesian updating"; Lighthill unlinked | First two fixed. Lighthill left unlinked: the name is used only for the term, and no new source was added. |

## 4. Targeted recheck (second fresh Opus sub-agent, read-only)

All items 1–7 and 9–18 were rated PASS. The reviewer independently re-derived:
- the sequential KL-control solution;
- the two-sided 3.2(d) induction bound, including the base case $t=T$.

It also verified the Byrnes §9.2 title online. Its structural sanity pass also passed:
- anchors;
- no answer cell starts with a heading;
- each placeholder appears once;
- 8/8 `<details>`;
- Python syntax by inspection;
- O(1)-sentence reading prose.

It raised five new nits, all fixed afterwards:
- the general posterior is the prior reweighted by $e^R$, and Boltzmann only in the deterministic case;
- 3.1(b) feasibility includes the start state;
- the 3.2(d) display now restricts to reachable $s'$, with a single global $p_{\min}$;
- the §4 sketch says "by some $\phi_i$ … that $U_i$";
- the Hubinger §2.1 line was folded into the blurb.

These last five fixes were checked by the owner only, not by a third critic.

## 5. Validation method and limitations

- **No shell was available.** `python build_wb1.py --check` was **not run**. Validation was by hand, by the owner after each edit round and independently by the recheck agent:
  - every `](#anchor)` has a matching `<a id>`, and nothing targets `#wb1-section-6`;
  - there are no local links;
  - no answer cell starts with a heading;
  - each `fill()` placeholder appears exactly once, and no LaTeX matches the leftover pattern;
  - `<details>` tags are balanced;
  - Python syntax checked by inspection.

  The root should run `cd notebooks && python build_wb1.py --check`. Expect "stale", but an AssertionError or SyntaxError would need fixing. Then regenerate the notebooks.
- Cross-notebook references are at main-section level: WB2, WB2 §2, WB3 §1, WB3 §6, WB4, WB5, WB5 §1. They were checked against the agreed structure and against the current `build_wb3.py` (which cites WB1 §3, §5 and Exercise 3.5). Other builders were being edited concurrently.
- Source checks:
  - Levine: equation and section numbers checked by the owner from the PDF.
  - Sutton, Hubinger, Retarget and Byrnes: checked by the critics. Their claims are recorded as reported; the owner did not re-check them.
  - The Wentworth NAH blurb uses the wording `reading-format-check.md` proposed. The owner did not re-check it against the post.
- Breadth additions (finding 8) are left for CRW.

## 6. Anchor and title changes others should know

- `wb1-section-6` ("What survives?") is **deleted**. Sections are now §1–5, with anchors `wb1-section-1` to `wb1-section-5` unchanged.
- §5.D is deleted. Cite "WB1 §5" for the simulator pointer.
- §4 subsections (4.A–4.C) are gone. §4 is now pitch, reading, setup and Q4.1.
- Exercise 3.5 is retitled "KL-regularized control as a Bayesian update"; its numbering is unchanged.
- §3.C is retitled "Where this leaves statistical learning".
- The Bellman optimality equation moved from §1 to §3.A.
- Q5.1 and Q6.1 are deleted. The questions are now 1.1, 2.1, 3.1–3.5 and 4.1.
