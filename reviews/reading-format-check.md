# Reading-format check: WB1, WB3, WB4 (2026-10-08)

Independent editorial audit by Claude Opus 5.5 (high effort). It covers the current `notebooks/build_wb1.py`, `build_wb3.py` and `build_wb4.py`, read while their owners were still editing them. The live edits I saw during the read were small technical fixes, so the findings below should still apply. I cite sections by heading because line numbers keep moving. I did not edit any builder. Technical and source repairs are left to `reviews/technical-sources.md`.

## The test applied

The brief, and the instructor's answer it records, set this rule: a reading-focused part gets **O(1) sentences on what the reading shows, plus the link**. Substantial text belongs only to derivation setup, exercises, hints and worked answers. I sorted each block into one of three kinds:

- **Necessary setup.** Definitions, equations or statements that a following exercise uses. Keep these, even when they are long.
- **Reading blurb.** A `reading()` call or an equivalent of 1–3 sentences. Keep these.
- **Narration.** Prose that explains a reading's ideas, history or argument without being used by an exercise. This is the v2 habit of "prose narrates ideas, not papers", and it should go. A short blurb that follows a long mini-essay still counts as narration.

I found no attributed instructor views, schedules, convergence cards or "predict before opening" instructions in any of the three builders. Solutions are stated as open to everyone.

## Top findings

1. **The derivation sections fit well. Keep them as they are**: WB1 §3, WB3 §3.B, §4.1–4.2, §5.B and §6.C, WB4 §2, §3.A–3.B, §3.C.i and §4.B.i. They link the source, give real setup and hints, and have full worked answers.
2. **Most reading-focused sections are still mini-essays with a blurb at the end.** The worst cases are WB1 §4 (about 700 words before Wentworth's blurb, which then says "the three claims above"), WB3 §2 (about 450 words, although the route calls it "reading only; the shortest section") and WB3 §6.B and §6.D. Also WB4 §4.A and §4.B (about 650 words narrating Furman and the Hänni–Vaintrob slides, with neither needed by Exercise 4.B.i) and WB4 §5.B (the RHM results listed as bullets, then repeated in the blurb).
3. **Readings often sit after the essay that replaces them.** This applies to WB3 §4.C (the GDL and Vaswani blurbs come after the hardware-lottery prose), §5.C (Hoffmann) and §6.D (janus and Nardo come after §6.A–B), and to WB4 §3.D, §4.C and §5.B. The fix is to move each blurb to the top of its subsection and delete the prose it duplicates.
4. **The generic claim/evidence apparatus survives in WB4 and in the WB1 closers.** WB4 §1.B is a four-item rubric (space, mechanism, estimand, "differential prediction"), and §1.B.i, §3.C.ii, the epistemic tags in §4.A and §6.B all apply it. WB1 has Question 5.1 ("case for relocates / case for dissolves / what would decide it") and §6's intact/transformed/dissolved table. These are the questions that exist only to preserve earlier content.
5. **WB4 still uses Bayes ensemble / Gibbs / single network as a unifying theme.** It appears in §1.B item 3, the "Estimand" bullets in §3.B, §4.B and §6.B, and the Gibbs caution in §3.D. Keep it only where a theorem's statement needs it: Solomonoff is a mixture bound, and the circuit-prior theorem bounds the posterior's average probability on the correct label.
6. **Contested theses appear under a "The result." label.** See WB1 §5, WB3 §4 and WB3 §6, detailed below. In reading-focused sections, replace the "**The result.**" label with a one-sentence pitch that does not take sides. "The result" fits derivation sections, where something is actually shown.

---

## WB1

| Section | Verdict | Action |
|---|---|---|
| Intro / route | Fits | Update the route after the cuts below: §6 goes, and §4 becomes "short, one core reading". |
| §1 Three pictures | **Narration** (about 400 words, no reading) | See below. |
| Q1.1 | Acceptable, optional | It is a specific taxonomy question. Keep it if §1 survives as a table. |
| §2 Why GOFAI failed | **Narration** (§2.A–2.B, about 300 words) | See below. |
| Q2.1 | Acceptable | It is tied to Sutton's text. Delete the last answer bullet ("What a real counterexample would look like…"), which is falsification-rubric boilerplate. |
| §3 Planning as inference | **Fits** | §3.A is necessary setup and Exercises 3.1–3.5 are real derivations. §3.C (about 180 words, no source) should shrink to 2 sentences or move into "What carries forward". |
| §4 Alignment, then | **Narration** (about 700 words) | See below. |
| §5 Alignment, now | Blurbs fit; the framing does not | See below. |
| Q5.1 | **Generic discussion; delete** | Its answer follows the case-for / case-against / what-would-decide template. |
| §6 What survives? | **Delete** | A synthesis question that preserves §4–5 content: a rubric table of verdicts and about 450 words of pro and con positions. Hubinger §4 is already in Further study. |

**§1 → a table.** Replace the four paragraphs with one sentence and a 3-row table with columns *picture | formal object | written by hand*:

- search: state space, operators, heuristic $h$; written by hand: $S$, operators, often $h$
- control: MDP $(S,A,P,r,\gamma)$; written by hand: reward, Markov state
- learning: $\mathcal H,\ell,\mathcal D$ and ERM; written by hand: $\mathcal H$

Exercises 3.2(d) and 3.5(b) cite "§1's Bellman equation", so move the Bellman optimality equation into §3.A as one displayed line, or keep it under the table. Drop the A* paragraph and "The pictures mix" (Q1.1 covers it).

**§2 → 2 sentences + Sutton.** Delete §2.A, §2.B and the "The result" paragraph. Replace them with: *"A search tree with branching $b$ has $\sim b^d$ lines of play; a tabular value function over $k$ variables at $m$ levels has $m^k$ entries. Lighthill's 'combinatorial explosion' and Bellman's 'curse of dimensionality' are the same blowup, and hand-written structure beats it only one problem at a time."* Then keep the Sutton blurb as it is.

**§4 → pitch, sketch, Wentworth.** Keep:

- One pitch sentence, replacing "The result": *"Classical alignment meant writing a utility over the agent's world model; the ontology problem is that the model can change, leaving the old utility with several extensions."*
- The "two-line sketch", including the one sentence on $\arg\max$ seeking the ambiguous region $A$. This is formal setup that Q4.1 uses. Use the diamond as a single example clause inside it.
- The Wentworth blurb, moved here with its result rewritten so it does not point back at deleted text: *"Proposes the natural abstraction hypothesis: information about a system that matters far away has a low-dimensional summary; humans use roughly those summaries; and diverse learners converge on them. Argues alignment becomes much easier if it holds, and plans simulation tests."*
- Q4.1.

Delete §4.A (von Neumann–Morgenstern, Dutch books, Goodhart, instrumental convergence and deception, all unsourced and used only by §6), the diamond narrative paragraph and §4.C.

**§5.** Replace the "The result" paragraph with *"Three readings on what a learned agent's values are; WB3 §6 adds a fourth (simulators)."* The current text states "this *relocates* the classical problems rather than removing them" as a result, which is a contested thesis that also pre-answers Q5.1. Keep the §5.A–5.C blurbs and their one-line cross-links. Cut §5.D to one line: *"Simulators and personae: see WB3 §6 (janus; Nardo)."*

## WB3

| Section | Verdict | Action |
|---|---|---|
| Intro / route | Fits | Once §2 is fixed, its "Reading only; the shortest section" label will be true. |
| §1 | **§1.A narrates the cake**, then the blurb restates it | Delete §1.A. Move the three slide quotes into Exercise 1.1(d) as one sentence: *LeCun's slide credits RL with "a few bits for some samples", supervision with "10→10,000 bits per sample" and prediction with "millions of bits per sample".* Delete "What the count does not show". The result line and answer (b) now carry that caveat. |
| Exercise 1.1 | Fits | |
| §2 | **Narration** (§2.A–2.C) | See below. |
| §3 | **Fits** | §3.A is a short mathematical remark and §3.C is a 3-sentence bridge. One gap: the arithmetic-coding derivation links no source. Promote Delétang et al. (already linked in §3.D) to a `reading()` beside §3.B, pointing to its background section on arithmetic coding. |
| §4.A, Exercise 4.1 | Fits (necessary setup) | |
| §4.B | Table acceptable | Move the GDL blurb here from the end of §4.C. |
| §4.C | **Narration** (about 200 words) | Keep the complexity table. It is compact data, and the Vaswani blurb already points to the paper's version of it. Replace the prose with the Vaswani blurb, moved here, and turn Hooker into a reading: **[The hardware lottery](https://arxiv.org/abs/2009.06489)**, Sara Hooker, 2020 · *optional*: *"Argues that research ideas often win because they suit the available hardware and software rather than because they are better."* Delete the Mamba sentence, which is unsourced, or reduce it to "see WB5 §1". |
| §5.A | **Narration** of the Kaplan → Chinchilla history | Replace it with two blurbs at the top of §5. **Kaplan et al. 2020** (existing link): *"Fits power laws in model size, data and compute; concludes most extra compute should go to parameters ($N_{\rm opt}\propto C^{0.73}$)."* **Hoffmann et al. 2022**: move the existing blurb up from §5.C. Exercise 5.2 already contains the learning-rate-schedule detail it needs. |
| Exercises 5.1–5.3 | Fit | |
| §5.C | **Delete** | It repeats Exercise 5.3's answer and adds unsourced claims ("many recent open models…"). At most, keep one sentence on the data limit. |
| §6 result line | **View as result** | See below. |
| §6.A | Partly necessary | Keep the persona-mixture equation and the sentence on prompting as conditioning, which Exercise 6.2 uses. Replace the janus paragraph with the janus blurb, moved up. |
| §6.B | **Narration of Nardo** (about 200 words) | Replace it with the Nardo blurb, moved up. Move the one sentence Exercise 6.2(c) needs (the evidence asymmetry and attractor claim) into the 6.2(c) prompt, and change the answer's "§6.B" to "Nardo's attractor argument". |
| §6.C, Exercises 6.1–6.2 | Fit | Trim "The caveats are real" to one sentence. Drop the last sentence of answer 6.2(c) ("Measuring how often… would bear on…"), which is an evidence-design tail. |
| §6.D | **Narration** (about 300 words) | Replace it with the RL's Razor blurb and split Mukherjee out of that blurb's "about" into its own optional `reading()` (existing link): *"RL fine-tuning updates are sparse (5–30% of parameters) yet nearly full-rank."* Delete the "selection vs search" paragraph and the pass@k test. Both are unsourced and contested. |

**§2 → 3 sentences + LeCun.** Delete §2.A–2.C and the "The result" paragraph. Replace them with: *"Unsettled. For text: nearly every bit was put there by a mind, so predicting it may force a model to model minds (janus's 'Simulators', §6, is the fullest statement of this side). For video: far more bits, grounded in physics. LeCun argues that the right objective is then to predict learned representations rather than pixels."* Then keep the LeCun blurb, trimming "read it as the strongest available statement of the counter-position", which the pitch now covers. The weak-versus-strong distinction and the "follow the proof" claim are unattributed views. Drop them.

**Views stated as results.** In §6, "a pretrained language model is better described as a simulator … than as an agent" and "'RLHF selects a persona' is a fair description" are janus's and Nardo's theses presented as results. Replace them with: *"Base models read as simulators of personae (janus), and what KL-regularized RLHF provably does to such a mixture (two exercises)."* In §4, "symmetry does not explain why the transformer won… that matters at least as much" is Hooker's thesis. Attribute it or cut it to "symmetry, and the hardware lottery".

## WB4

| Section | Verdict | Action |
|---|---|---|
| Intro / route | Fits | Update the route for the §1 and §6 changes. |
| §1 | **Delete the rubric** | See below. |
| §2 (all) | **Fits** | This is the model for the format. One small point: the last sentence of §2.A on reparametrization invariance gives away Exercise 2.D.ii. Cut it. |
| §3.A–3.B, Exercise 3.B.i | Fit | Fold the "Estimand" remark into a clause of the theorem paragraph ("a bound on the mixture forecaster's cumulative log-loss, not on any single hypothesis"). Keep the Bayes-mixture remark. |
| §3.C | First four bullets fit; **the Legg bullet narrates Legg** (about 150 words) | Replace it with a Legg `reading()` placed at Exercise 3.C.i, moved here from Further study (see below). |
| Exercise 3.C.i | Fits | |
| Exercise 3.C.ii | **Generic discussion; delete** | "Classify each step as definition / theorem / empirical claim" is the claim/evidence rubric, and its answer duplicates §3.D. |
| §3.D | R&H blurb fits; **the bridge paragraph narrates Valle-Pérez** (about 170 words) | See below. |
| §4.A–4.B | **Narration** (about 650 words) | See below. |
| Exercise 4.B.i | Fits | It is self-contained. Re-point the answer (c) references "link 3 of §4.A" and "(A)/(B)/(C) of §6.A" to "the search problem Furman names" and the §6 table. |
| §5.A | Mini-derivation with no source and no exercise | Cut it to one sentence: *"A depth-matched network approximates a binary tree of 2-variable functions with $\sim(d-1)\varepsilon^{-2/r}$ units versus $\varepsilon^{-d/r}$ for a shallow one (an existence result: Furman's 'paradox of approximation')."* |
| §5.B | **Results narrated, then repeated in the blurb** | Move the RHM blurb to the top. Keep the three model-ingredient bullets and the line "$P^*\simeq n_cm^L$ for deep CNNs", which Exercise 5.B.i uses. Delete the "reported results" bullets and the interpretive paragraph. That paragraph also gives away Exercise 5.B.i's answer. |
| Exercise 5.B.i | Fits | |
| §5.C | Unsourced counting essay | Cut it to one sentence pointing to WB5, or delete it. |
| §6 | **Rubric/summary essay** | See below. |

**§1.** The section contains no reading and no derivation. Its framework (four things every mechanism must name, including a "differential prediction" and the three-estimand trichotomy), plus Exercise 1.B.i, is a universal evaluation rubric, which the instructor rejected. Merge it into the intro as 2–3 sentences: *"By no-free-lunch (WB2 §2), a learner that does well on the tasks we meet must favour some targets over others; 'inductive bias' names that preference without saying what it is. §2–5 give candidate answers: prior volume, description length, circuit size, kernel norm, and structure in the data."* Delete §1.A, §1.B and §1.B.i.

**§3.C, the Legg blurb:** **[Is there an elegant universal theory of prediction?](https://arxiv.org/abs/cs/0606070)**, Shane Legg, 2006 · *core for 3.C.i*. *"For computable predictors of computable sequences: every predictor is defeated by a sequence of about its own complexity; predictors for all sequences up to complexity $n$ exist but must themselves have complexity about $n$; and beyond some $n$ no such predictor can be proved to work. Short, with each proof about a page; Exercise 3.C.i follows Lemmas 3.2–3.3 and Theorem 4.3."*

**§3.D.** Delete the bridge paragraph and its Gibbs-estimand caution. Append one sentence to the Valle-Pérez blurb's "about": *"The bound is one-sided and uses a computable proxy for $K$; §7 tests whether SGD samples like the posterior."* Label it *optional*.

**§4.A–4.B → two blurbs.** Exercise 4.B.i does not depend on any of the surrounding narration. Open §4 with the Furman and Hänni–Vaintrob blurbs, moved up from §4.C; both are already good. Add to the Hänni–Vaintrob "about": *"Slide 34 contrasts this scaling with NNGP, which it describes as needing $\sim d^k$ samples for $k$-sparse parity; Exercise 4.B.i derives both counts."* Delete:

- the link-by-link chain with its *Empirical*/*Theorem* tags
- the Furman evidence paragraph
- the circuit-prior setting bullets, the theorem restatement and "Three things to notice"

The blurb already states the theorem, and the "Estimand" bullet is the Gibbs theme.

**§6 → one table.** Replace §6.A–6.C with one table: *prior (program length / circuit size / kernel norm) | where in this notebook | sample cost on $k$-sparse parity | correlated vs uncorrelated RHM*. Add one line on SLT: *"SLT describes posterior concentration in parameter space (§2.A's volume, done correctly for singular models), not a fourth prior; see the SLT day."* Delete the "Estimand" and "Width at fixed data" bullets.

## Not in scope here

Mathematical correctness, source claims and quoted figures are covered by `reviews/technical-sources.md`. If these cuts remove a block that a technical fix targeted, the fix becomes moot. Check that no cross-reference into a deleted block survives, for example WB1 §6 → §4.A, WB4 Exercise 4.B.i(c) → §6.A, and WB3 Exercise 6.2(c) → §6.B.
