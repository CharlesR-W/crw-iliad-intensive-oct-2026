# WB2 editorial (design/organization) review — 2026-10-08

Reviewer: Claude Opus 5.5 (high effort), read-only on builders. Scope: organization, scaffolding, navigation, framing. Technical correctness is in `reviews/wb2-technical.md` (Codex). Authority: `reviews/2026-10-08-editorial-brief.md`.

Sources reviewed: `notebooks/nbkit.py`, `notebooks/build_wb2.py`, `notebooks/wb2_cells_a.py` (opening, §1–§5), `notebooks/wb2_cells_b.py` (§6–§9, closing). Anchors below are the section heading plus a quoted snippet, not line numbers. In multi-line quotes, line breaks follow the current source. Integrators should match the text and ignore differences in wrapping.

## Summary

- WB2's core is sound. It is a derivation workbook in which almost every derivation is already an exercise with setup, a worked solution, and usually a hint. No derivation is lectured in prose where an exercise would teach better. The readings are short (2–4 sentences plus a section guide). Keep all of this.
- The problems are scaffolding at the two ends. The title page has a "Main results" list, a schedule-like "Time" block, a prediction-gate instruction, and a "Three estimands" block that makes Bayes/Gibbs/MLE a rubric across the whole notebook. The closing cell has "What carries forward", "Further study", and a "Convergence card" tied to "the 17:00 convergence".
- There are four attributed or ghost items: "CRW's position" (§6.A), "closest … to CRW's position" (§6.F), "CRW's Morse–Bott note" with no content (§8.D), and a "supplied sweep table (to be added)" that does not exist (§9.B).
- There are five prediction gates: the title page, §4.C, §5.C, §8.C.iii and §9.B.
- Smaller items: a commit-then-revise discussion pair wrapped around the §6 derivations, one duplicated question in §2, a padding paragraph after §2.A, two "Discussion" labels, six harder exercises with no hint, and one cited paper with no link.
- The section numbers §1–§9 and the anchors `wb2-section-1`…`wb2-section-9` must not change. WB1, WB3, WB4, WB5 and WB6 cite WB2 by section number.

## Recommendations (priority order)

### R1. Title page: replace "Main results", "Route", "Time" and "How to use" with one route that has short descriptions

**Location.** Opening cell (`# Workbook 2 — Statistical learning and regularization…`), from `**Main results**` through `…so commit before\nyou open them.` The `**Notation, used throughout.**` block that follows stays.

**Action.** Replace that whole span with:

```markdown
**Route.** Start with §1 (gap vs excess risk), which later sections rely on.
After that, pick whatever interests you; nobody is expected to do all of it.
§5 uses §1.B, §4.A and §4.C, and §8 and §9 read best after §7. Parts marked
*optional* can be skipped without losing the thread.

- [1. Risks and gaps](#wb2-section-1): the generalization gap vs the excess risk, both computed exactly for least squares.
- [2. Doesn't some theorem make this impossible? (天无绝人之路)](#wb2-section-2): no free lunch, universal approximation (with a depth-separation exercise), and why elementary class-wide capacity bounds are vacuous at realistic network sizes.
- [3. Bias–variance](#wb2-section-3): the pointwise decomposition, and ridge regression direction by direction.
- [4. Regularization three ways, and how not to think about it](#wb2-section-4): ridge as prior, constraint and spectral filter; early stopping; dropout (optional); when flatness can matter and why raw sharpness cannot.
- [5. Overfitting and double descent](#wb2-section-5): the least-squares double-descent curve, derived, and why optimal ridge removes the peak.
- [6. Information bottleneck and the information plane](#wb2-section-6): why mutual information is degenerate for deterministic networks, what binned estimators measure, and compression as clustering in a noisy model.
- [7. Bayes and MLE generalization error](#wb2-section-7): the sandwich covariance, TIC, AIC and BIC, and the errors of a single fit, the Bayes ensemble and a posterior draw.
- [8. Does it cancel? $C$ vs $H$](#wb2-section-8): gradient noise vs curvature, and whether SGD's stationary distribution looks like a posterior.
- [9. Fantastic generalization measures](#wb2-section-9): which published measures predict the generalization gap in a large sweep, read against §7–§8.

**Using the notebook.** Work on paper or in the blank answer cells. Hints are
folded under "Hint". The solutions notebook, which is available to everyone,
has a worked solution for every problem.
```

**Reason.** This removes the duplicated "Main results" list, the time estimates, the "for your own notes" framing and the prediction-gate instruction. It gives self-directed students one route with a short description of each section, and it states the real dependencies. The previous text said the sections were independent, but §5, §8 and §9 are not. The formula-level results stay in each section's **The result.** paragraph.

### R2. Title page: delete the "Three estimands" block (a Bayes/Gibbs/MLE rubric across the notebook)

**Location.** Opening cell, from `**Three estimands.** "Generalizes well" can mean three different things:` through `almost entirely about the first, and every section says which one it means.`

**Action.** Delete. Then make the one dependent edit in §3.A.i.3. Replace

> `Translate this into a statement about the Gibbs and Bayes-ensemble estimands.`

with

> `Translate this into a statement comparing a single posterior draw (the *Gibbs* predictor) with the posterior-predictive mean (the *Bayes ensemble*); §7.E defines both errors carefully.`

**Reason.** "Every section says which one it means" makes the estimand distinction a notebook-wide rubric. The brief allows the distinction locally, and §7.E defines it fully. §3.A.i.3 is the only place before §7 that relies on the title-page definitions.

### R3. Closing cell: delete "What carries forward" and "Convergence card"; move the "Further study" items into the sections they serve

**Location.** The last markdown cell, starting `## What carries forward`. It also contains `## Further study` and `## Convergence card` / `Bring one card to the 17:00 convergence.`. The answer cell after it starts `An example card, for the shape rather than the content:`.

**Action.**
1. Delete the whole closing markdown cell and the example-card answer cell.
2. Move the Further study items into the sections:
   - **§7.E.** Replace `(Watanabe defines them without the outer $\mathbb E_D$ and states results for their expectations.)` with `(Watanabe, *Mathematical Theory of Bayesian Statistics*, 2018, ch. 1–2, defines them without the outer $\mathbb E_D$, states results for their expectations, and treats the singular case and WAIC.)`
   - **§7.A.i answer.** Replace `(Huber; White 1982 for misspecified likelihoods)` with `(Huber; H. White, "Maximum likelihood estimation of misspecified models", *Econometrica* 1982, for misspecified likelihoods)`
   - **§6.F Readings.** In the `+ r"""` string that follows the Goldfeld et al. `reading(...)`, before `#### 6.F.i`, add one line: `Background: T. Cover and J. Thomas, *Elements of Information Theory*, ch. 8–10 (differential entropy, the Gaussian channel, rate–distortion).`
3. Keep the one cross-notebook pointer that exists nowhere else, as an inline sentence (see R14 for its location): `WB6 §9 (gradient descent in a tiny subspace) and §10 (spectra and curvature) look at the measured structure of $H$ in real networks.` I am not adding the old "§6 → WB5 / Platonic Representation Hypothesis" pointer: I could not confirm that WB5 takes it up. WB6 already points back to WB2 §6.

**Reason.** The brief removes convergence cards and shared schedules ("17:00"). "What carries forward" is cross-notebook scaffolding. The forward links that teach something are already inline: WB4 §2 in the §7.D answer, WB5 in §4.C and §5.E, and Training Dynamics in the §8.C.iii answer. The further-study items are more useful next to the material they support.

### R4. §6: remove the attributed instructor view and the references that depend on it

**Location and actions.**
1. **§6.A.** Replace `**A third reading** (closest in the literature: Goldfeld et al. 2019). Add` with `**A third reading** (Goldfeld et al. 2019). Add`.
2. **§6.A.** Delete the paragraph from `**CRW's position**, stated as a position for you to test rather than a` through `happening, and it should be measured geometrically rather than\ninformation-theoretically.`, and the `+ commentary("CRW's position on the information plane, in his own words")` that follows it.
3. **§6.F Goldfeld reading, `about` string.** Replace `"worked examples. The estimator's convergence theory can be skipped. This is the paper closest "` and `"to CRW's position.",` with the single string `"worked examples. The estimator's convergence theory can be skipped.",`.
4. **§6.D.ii.4 answer, last sentence.** Replace `Whether that makes mutual information "the wrong\nfunctional" or "the right functional, with a scale" is the question 6.A left\nopen.` with `Whether that makes mutual information the wrong quantity, or the right one at a stated scale, is position (iii) of 6.F.i.` If R5 is not applied (fallback), use `…at a stated scale, is position (iii) of 6.A.i.` instead.

**Reason.** The brief says to remove attributed instructor views, and the instructor will add his own. Items 1, 3 and 4 refer to the deleted paragraph and would otherwise become ghost references.

### R5. §6: merge the pre-derivation "commit now" discussion (6.A.i) into the closing question (6.F.i)

**Location.** In the §6.A markdown cell, the block from `#### 6.A.i. Before the derivations: what would discriminate?` through `Commit now;\nyou can revise after 6.D.`. Its answer cell starts `A good answer separates three pairs.`. The later block starts `#### 6.F.i. Back to the three accounts` and has the answer `A good answer maps each derivation to a claim, without needing to declare a\nwinner.`

**Action.**
1. Delete the 6.A.i block from the §6.A cell. The cell should end at `The binned estimates track the same clustering.`
2. Delete the 6.A.i answer cell.
3. Replace the 6.F.i block (`#### 6.F.i. Back to the three accounts` … `Has any of your answers changed?`) with:

```markdown
#### 6.F.i. What would discriminate the three accounts?

Three positions: (i) compression in mutual information is real and causes
generalization; (ii) it is an artefact of saturation and binning, with no
causal link to generalization; (iii) it is real geometric clustering,
mismeasured as mutual information.

1. What does each of 6.B, 6.C and 6.D establish about these positions?
2. For each pair, describe an observation or intervention that would favour one over the other.
```

4. Replace the 6.F.i answer cell with two parts:
   - `**1.**` followed by the current 6.F.i bullets. Drop the lead sentence `A good answer maps each derivation to a claim, without needing to declare a winner.` Change `which the\n  invariance test in 6.A.i addresses` to `which the invariance test in part 2 below addresses`, and `which needs the interventions from 6.A.i.` to `which needs the interventions in part 2 below.`
   - `**2.**` followed by the current 6.A.i bullets, verbatim from `- **(i) vs (ii).**` through the end. Drop the lead `A good answer separates three pairs. (These are the kinds of evidence that\ndiscriminate; this cell does not pick a winner.)`

**Reason.** A "commit now, revise later" claim/evidence prompt before the derivations is the generic padding the brief targets. The content of both answers is specific and worth keeping. One question placed after the material keeps all of it and removes the wrapper.

**Fallback.** If root prefers a minimal edit, keep both cells, but delete `Commit now;\nyou can revise after 6.D.` and `Has any of your answers changed?`.

### R6. §8.D: delete the ghost "Morse–Bott note"

**Location.** In the §8.D `Readings and rabbit holes` cell: `+ "\n\n**CRW's Morse–Bott note** (on posterior concentration near degenerate minima; rabbit hole).\n\n"` and `+ commentary("CRW's Morse–Bott note: what it says and where to find it")`.

**Action.** Delete both. The cell ends after the Hänni–Vaintrob `reading(...)`.

**Reason.** As built, students see a bold heading with no content and no link. It is also an attributed item. The instructor can add it himself if he wants it (see Q1).

### R7. §9.B: replace the prediction exercise built on a missing table with a reading-and-reasoning question

**Location.** The cell starting `### 9.B. Commit, then check` / `The supplied sweep table (to be added) records`. Its answer starts `*Pending: the sweep table has not been added yet`.

**Action.** Replace the markdown cell with:

```markdown
### 9.B. Five measures, read against §7–§8

#### 9.B.i. Question

For each of these five measures, say what Jiang et al. found (where they
include it), and use §7–§8 to say which hyperparameter axes (width, depth,
learning rate, batch size, weight decay) you would expect to drive its
correlation with the gap:

1. parameter $\ell_2$ norm;
2. product of layer spectral norms over margin²;
3. PAC-Bayes (magnitude-aware) sharpness;
4. final gradient variance $\mathrm{Tr}\,\hat C$;
5. TIC, $\frac1n\mathrm{Tr}\big((\hat H+\epsilon I)^{-1}\hat C\big)$.
```

In the answer cell, delete the italic `*Pending: … with Jiang et al.*` paragraph. Replace `Reasoning a good prediction would contain, and what Jiang et al. found for\nthe measures they share with this list:` with `What Jiang et al. found for the measures they share with this list, and what §7–§8 suggest:`. Keep the five bullets unchanged.

**Reason.** The table does not exist. The brief rules out training and paid compute, so the current cell is both a ghost reference and a prediction gate. The existing worked answer already supports the reworded question.

### R8. Remove the remaining prediction gates (§4.C, §5.C, §8.C.iii)

**§4.C.** In the cell before the `spectral_filters.html` figure, delete the paragraph from `**Commit before you look.** On axes of $\log s^2$ against filter value, sketch` through `where the two differ most and which one is on top there.` Keep the `figure(...)` call. The comparison is already exercise 4.C.i.3, which has a full answer.

**§5.C.** Replace the cell
```
**Commit before you look.** Sketch the min-norm risk against $\gamma$ on a log
axis, once for $r^2/\sigma^2=5$ and once for $r^2/\sigma^2=0.5$. Draw the null
risk $r^2$ as a horizontal line. Shade the bias and variance contributions.
Then sketch optimal ridge on the same axes.
```
with
```markdown
#### 5.C.ii. Sketch the curves

Sketch the min-norm risk against $\gamma$ on a log axis, once for
$r^2/\sigma^2=5$ and once for $r^2/\sigma^2=0.5$. Draw the null risk $r^2$ as a
horizontal line. Shade the bias and variance contributions. Then sketch optimal
ridge on the same axes. The interactive figure below draws all of these.
```
Keep the answer cell (`*What a correct sketch shows.*`) and the figure. This is a real consolidation exercise, so only the gate goes.

**§8.C.iii.** Replace
```
#### 8.C.iii. Commit, then look

Take $H=\mathrm{diag}(1,10)$: $\theta_1$ is flat and $\theta_2$ sharp. Before
opening the figure, sketch three pairs of 2σ ellipses, SGD's $\Sigma$ and the
posterior's, for $\gamma=0,1,2$. Which way does each SGD ellipse point? What
do you expect to happen if $C$'s eigenbasis is rotated relative to $H$'s?
```
with
```markdown
#### 8.C.iii. Sketch the ellipses

Take $H=\mathrm{diag}(1,10)$: $\theta_1$ is flat and $\theta_2$ sharp. Sketch
three pairs of 2σ ellipses, SGD's $\Sigma$ and the posterior's, for
$\gamma=0,1,2$. Which way does each SGD ellipse point? What happens if $C$'s
eigenbasis is rotated relative to $H$'s? The figure below draws them.
```

**Reason.** The brief removes instructions to predict before opening figures. The title-page gate goes in R1, and the `figure()` helper text is being changed by root in `nbkit.py`.

### R9. §2.A: delete the paragraph that repeats the point after NFL

**Location.** The standalone cell `NFL is often cited as showing that "there is no universally good learner".\nThe useful reading is the reverse: every learner that works is making a bet\nabout the targets, and the interesting question is which bet.`, together with its `+ commentary("what prior over targets the world has…")`.

**Action.** Delete the whole cell.

**Reason.** It repeats §2's **The result.** ("the way out is a bet on structure in the targets") and the 2.A.i.4 solution almost word for word.

### R10. Relabel the "Discussion" headings as "Question"

- Replace `#### 7.F.i. Discussion` with `#### 7.F.i. Question`.
- Replace `#### 9.C.i. Discussion` with `#### 9.C.i. Question`.

**Reason.** Both prompts are specific and have substantive worked answers, so keep them. The label "Discussion" signals generic discussion.

### R11. Add hints to six harder exercises that have none

Append each hint at the end of the named exercise's markdown cell, before its answer cell.

- **4.B.i** (cell ending `Which objection from §2 applies to that explanation?`):
```markdown
<details><summary>Hint</summary>

For 1, write the Lagrangian $\hat R_n(b)+\lambda(\|b\|^2-c)$ with $\lambda\ge0$, set its gradient to zero, and use complementary slackness. For 2, write $\|\hat\beta_\lambda\|^2$ in the SVD of $X$ and look at each term as a function of $\lambda$. For 3, recall §1.A.i.3 and §2.C.
</details>
```
- **4.C.i** (cell ending `What is the limit from a nonzero initialization $\beta_0$?`):
```markdown
<details><summary>Hint</summary>

Work in the coordinates $b_k=v_k^\top\beta$, where $X^\top X$ is diagonal and $X^\top y=\sum_ks_k(u_k^\top y)v_k$; gradient flow then decouples into one scalar linear ODE per $k$. For 3, set each filter equal to $\tfrac12$ and solve for $s^2$. For 4, in which subspace does $\dot\beta$ always lie?
</details>
```
- **7.B.ii** (the §7.B cell ending `By what factor is the naive answer wrong?`):
```markdown
<details><summary>Hint</summary>

At $\theta^*=\beta$ the per-sample gradient is $-\varepsilon x$. For a standard normal, $\mathbb E x^4=3$.
</details>
```
- **8.C.ii** (cell ending `Is $\mathrm{KL}^*$ symmetric in $\gamma$ for $p=3$?`):
```markdown
<details><summary>Hint</summary>

For zero-mean Gaussians, $\mathrm{KL}(\mathcal N(0,\Sigma)\,\|\,\mathcal N(0,S))=\tfrac12\big[\mathrm{Tr}(S^{-1}\Sigma)-p+\log\det S-\log\det\Sigma\big]$. Minimize over $\log T$, then write the result in terms of the eigenvalues of $H\Sigma$ and compare their arithmetic and geometric means. For 2, in the commuting case $H\Sigma\propto H^\gamma$.
</details>
```

- **6.D.i.** In the §6.D cell, insert before `#### 6.D.ii. A two-class model`. The existing hint in that cell covers only 6.D.ii.
```markdown
<details><summary>Hint</summary>

$h(T\mid X)=h(\sigma\varepsilon)$ is the same for every $f$. For 2, among all densities with a given covariance, the Gaussian has the largest differential entropy.
</details>
```
- **8.B.i** (cell ending `Does the envelope force a fitted $\gamma$ into $[1,2]$?`):
```markdown
<details><summary>Hint</summary>

For the lower bound in 2, use $\|v_i\|^2\ge v_{ij}^2$, then Jensen (or Cauchy–Schwarz) on $\mathbb E\,v_{ij}^4$. For 3, each example contributes to exactly one coordinate.
</details>
```

**Reason.** The brief asks for enough hints that students can learn without an LLM. These are the hint-less exercises where a student is most likely to get stuck. The others either state the method in the question (7.C.i, 7.D.i) or are short.

### R11b. §4.D: link the paper behind the dropout derivation

**Location.** The 4.D.i answer: `(This is the Wager–Wang–Liang observation.)`

**Action.** Replace with `(This is the observation of [Wager, Wang and Liang, "Dropout training as adaptive regularization" (NeurIPS 2013)](https://arxiv.org/abs/1307.1493).)`

**Reason.** The brief asks that each derivation link its original paper. The paper is already cited; only the link is missing. I confirmed the arXiv ID and title.

### R12. §2.C: drop exercise 2.C.i.4, which duplicates 2.D.i.1

**Location.** In the 2.C.i cell: `4. Which kinds of explanation does this argument leave standing?`. In its answer cell, the block from `**4.** The argument kills explanations of the form "the reachable class is small".` through `- the prior-over-targets answer of §2.A.4.`

**Action.** Delete both.

**Reason.** 2.D.i.1 ("List every explanation of generalization the random-labels experiment rules out, and every one it does not") asks the same question, and its worked answer is a superset.

**Lower priority.** Root may decide the other way: keep 2.C.i.4 and delete 2.D.i.1, so that students who skip the reading still see the point. If so, also fix the ghost cross-reference in the 2.C.i.4 answer: change `§2.A.4` to `§2.A.i.4`.

### R13. Source cleanup (no change to the rendered output)

- Remove the remaining `+ commentary(...)` calls. They return `""`, but their prompt strings carry attributed framing: "CRW's view on which of the three readings…", "CRW on whether any measure…", "which estimand matters … and for alignment", and others. They appear after §1, §3, §4, in §7.F and in §9.C. The calls after §2.A, in §6.A and in §8.D go with R9, R4 and R6.
- Remove the `note=` arguments that mention CRW: the Goldfeld note `"where this paper and CRW's position part ways, if anywhere"` and the Jiang note `"which of these results CRW thinks have held up"`. `reading()` ignores `note`, but future editors could misread them as live.

**Reason.** Prevents attributed framing from coming back in later edits.

### R14. Cross-day phrasing (minor; see Q3)

- In the 8.C.iii answer, replace `This is as far as WB2 goes. The Training Dynamics day takes up state-dependent` with `This is as far as WB2 goes. The Training Dynamics materials take up state-dependent`.
- At the end of the same paragraph, after `…what happens away from a quadratic\nbasin.`, add the pointer kept from R3: `WB6 §9 (gradient descent in a tiny subspace) and §10 (spectra and curvature) look at the measured structure of $H$ in real networks.` Use plain text, not a link. A link to another notebook with an `#anchor` would fail the local-link check in `nbkit._validate`.
- In the 7.D.i answer, replace `the subject of the SLT day.` with `the central quantity of singular learning theory (SLT).`

**Reason.** For students working through the notebooks on their own, a "day" reads as a schedule. The pointer is still useful without it.

### Technical items noticed in passing (for Codex, not editorial)

- **§8 "The result." has a garbled sentence**, likely from a concurrent edit: `$1\le\gamma\le2$. On a quadratic basin,\nIn the continuous-time approximation, SGD's stationary covariance solves`. Merge it into `On a quadratic basin, in the continuous-time approximation, SGD's stationary covariance solves`.
- **§8.C.i.5 sign contradiction: already fixed.** It now reads "increases with curvature".
- **§6.D.ii.3 answer has a redundant qualifier.** `In this model, compression in the noisy-network sense *is* within-class\nclustering in this restricted family.`
- **§2.C.i.2 answer is hard to read after the technical edits.** The sentence beginning `The rough VC scale $W\ln W\approx1.4\times10^7\gg n=5\times10^4$ likewise makes this generic upper-bound route uninformative (it does not prove…` could be split.

## Things to keep explicitly (do not over-cut)

- **Every derivation exercise and its worked solution.** That is 1.A.i, 1.B.i–ii, 2.A.i, 2.B.i, 2.C.i.1–3, 3.A.i, 3.B.i, 4.A.i–4.E.i, 5.A.i–5.C.i, 6.B.i–6.E.i, 7.A.i–7.E.i, 8.A.i–8.C.ii, plus the hints, the tables in 6.C/6.D/7.B.ii/8.C.i.5, and the pointers to `checks/wb2a_check.py` and `checks/wb2b_check.py`.
- **Section numbers and anchors.** Keep §1–§9 and `wb2-section-1…9` unchanged, and do not reorder sections. Other notebooks cite WB2 §2 and §4–§9 by number.
- **Each section's opening paragraph.** The **The result.** paragraph at the top of each section, and the "Setup, self-contained" blocks in §7–§9, are what let students enter a section without having worked through everything before it.
- **The title-page "Notation, used throughout." block** and the opening paragraph, including `Where a result is quoted rather than derived, the text says so.`
- **Local Bayes/MLE material.** Keep §3.A.i.3, §4.A, §5.C.i.2–3, all of §7 (including 7.E's estimands and 7.F.i), and §8.C.i.3/7. These teach the distinction where it is the subject.
- **Specific questions.** Keep 1.A.i.4, 2.A.i.4, 2.B.i.5, 2.D.i, 4.F.i, 5.D.i, 7.F.i, 9.A.i and 9.C.i.
- **Short bridge paragraphs.** Keep the end of §1 (`When someone says deep nets "generalize mysteriously", ask which quantity they\nmean.`), the end of §3 (`The textbook U-shaped curve…`) and the end of §4 (`**Where this leaves "the hacky bullshit".**`). Remove only their `commentary()` calls.
- **Optional labels.** Keep them on 4.D, 6.E, 8.C.ii and the 5.E reading, and the tier labels on readings: *core*, *optional*, *rabbit hole*.
- **All readings, including the circuit-prior rabbit hole.** Each is already O(1) sentences plus a section guide.
- **The three interactive figures**, minus their gate text, and the `Use panel A only (panel B belongs to WB5)` note.
- **Phrases from the topic sketch:** "why the hacky bullshit works" and 天无绝人之路.

## Open questions for root

1. **Morse–Bott note (R6).** Does the instructor have a link he wants under §8.D? If so, he can add it as a one-line `reading(...)`. Otherwise R6 deletes it.
2. **§9 sweep table (R7).** Confirm there is no plan to produce the sweep table. The brief rules out training. If a table is ever added, a measured-comparison exercise can return as 9.B.ii.
3. **"SLT day" / "Training Dynamics day" (R14).** WB4, WB5 and WB6 use the same phrasing. Root may want a single cross-notebook decision rather than changing WB2 alone.
4. **Further-study citations (R3).** Watanabe, White and Cover–Thomas have no free canonical links, so they are kept as plain citations. Is that acceptable? The independent checker suggested adding a freely available Watanabe paper for §7.E (for example, the 2010 JMLR WAIC paper). That would be a new paper, so it is left to the instructor. The same goes for links for Cybenko/Hornik (§2.B) and Takeuchi/Akaike/Schwarz (§7), which are quoted rather than derived.
5. **Instructor framing outside WB2.** `notebooks/README.md` calls `*_solutions.ipynb` the "instructor notebook", and `nbkit.answer()` documents the solution as "for instructors". Both conflict with "all solutions are available to all students". These files are outside WB2's builders, so the change is left to root.
6. **README entry for WB2 (suggested).** `02_statistical_learning`, labelled "Derivations: exercises with hints and worked solutions; a few core readings; three interactive figures". One-line description: "Why the hacky bullshit works: risks vs gaps, NFL/UAT/capacity bounds, bias–variance and ridge, regularization as prior, constraint and filter, double descent, the information plane, Bayes vs MLE generalization error, SGD noise vs curvature, and generalization measures."

## Independent check

A fresh, read-only Claude Opus sub-agent checked this file against the brief and the current WB2 source, which the technical revision is still changing. It found that every quoted snippet for R1–R14 locates in the source, that no recommendation contradicts "Things to keep", and that no new dangling anchors or heading-led answer cells are created. The grep sweep for commit/predict/CRW/instructor/discussion/day/time terms turned up nothing beyond what is covered here. The table lists each finding and what was done with it.

| Finding | Disposition |
|---|---|
| R1's "notation used throughout" was inaccurate, because the notation lives on the title page. §5 also uses §4.C. The §2 description overstated the source's hedge. | **Fixed** in the R1 text. |
| R1 "§8 and §9 build on §7" sat awkwardly next to the self-contained setups. | **Fixed**: now "read best after §7". The keep-list wording was adjusted to match. |
| The R4.4 fallback would point to a heading that no longer exists. | **Fixed**: the fallback now uses "position (iii) of 6.A.i". |
| R5: "in 2" was ambiguous. | **Fixed**: now "in part 2 below". |
| R8 §5.C.ii: "after the solution" reads oddly in the exercises notebook. | **Fixed**: now "The interactive figure below". |
| R3: the Cover–Thomas line had no clear insertion point. | **Fixed**: now specifies the `r"""` string after Goldfeld's `reading(...)`. |
| The §8.C.i.5 sign contradiction has already been fixed by the concurrent edit. | **Updated** the technical-items list. I confirmed it reads "increases with curvature". |
| Missed: a garbled sentence in the §8 **The result.** paragraph (`On a quadratic basin,\nIn the continuous-time…`). | **Added** to the technical items. I confirmed it in the source. |
| Missed: a cited paper with no link (Wager–Wang–Liang, §4.D). | **Added as R11b**. I confirmed the arXiv ID 1307.1493 and the title. |
| Suggested: add links or new papers for Watanabe, Cybenko/Hornik and Takeuchi/Akaike/Schwarz. | **Not adopted**: the task forbids new papers. Recorded in Q4. |
| Missed: no hints for 6.D.i and 8.B.i. | **Added** to R11. I did not add 3.A.i.4, which is short. |
| R3 drops the only WB6 navigation pointer. | **Adopted**: a plain-text pointer to WB6 §9–§10 is now in R3 and R14. I confirmed those WB6 section titles. Not adopted: the §6 → WB5 pointer, which I could not confirm. |
| Under the R12 alternative, the 2.C.i.4 answer has a ghost reference, `§2.A.4`. | **Added** to the R12 note. |
| The Zhang–Yang–Tu reading guide is borderline long, at about 5 sentences. | **Not adopted**: it is within "a few sentences" plus a section guide. The optional trim is left to root. |
| §6.D.ii.3 still has its redundant qualifier. | Already listed in the technical items. |
