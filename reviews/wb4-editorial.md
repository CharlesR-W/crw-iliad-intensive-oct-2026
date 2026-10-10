# WB4 editorial record, 2026-10-08

Owner: the Claude Opus 5.5 agent assigned to WB4 for this round. The only files edited were `notebooks/build_wb4.py` and this file. The previous owner's rewrite and math fixes were the starting point and were not reverted. The pre-rewrite snapshot (`/tmp/mdl-before-20261008/notebooks/build_wb4.py`) was used only for comparison.

**Tooling limitation.** Neither this agent nor any sub-agent had a shell tool in this session, so `python build_wb4.py --check` was **not run**. Its validator conditions (the `_validate` function in `nbkit.py`) were checked by static inspection three times: by me and by both critics. The generated `.ipynb` files were not touched. Someone with a shell should run `cd notebooks && python build_wb4.py --check`. The expected result is "stale: …", with no AssertionError.

## 1. Initial critique

This reconciles `reading-format-check.md` and `technical-sources.md` against the builder as it was when this round started.

**Already done:**

- Technical #5 (Solomonoff telescoping): $\xi(\epsilon)$ is in the product and the regret, and nonnegativity against the subprobability is proved in 3.B.i(c).
- The finite-function coding wording.
- No instructor views, schedules or prediction gates.
- Solutions are open to all.

**Remaining at start:**

- §1 was still the four-item rubric (space, mechanism, estimand, differential prediction), with Exercise 1.B.i and its answer.
- §2.A's last sentence gave away the evidence-invariance part of 2.D.ii.
- In §3.B, the "Estimand" bullet ran the Gibbs/ensemble theme as a separate remark.
- Technical recheck item 7 was not done: $K(\nu)$ was undefined for lower-semicomputable $\nu$.
- The §3.C Legg bullet was about 150 words of narration.
- Exercise 3.C.ii was rubric apparatus ("definition / theorem / empirical claim").
- §3.D had a bridge essay with a Gibbs caution, and the R&H core reading sat at the end of §3 rather than the top.
- §4.A–4.B were about 650 words: the link-by-link chain with *Empirical*/*Theorem* tags, Furman's evidence, the circuit-prior setting and "Three things to notice", including an Estimand bullet. The blurbs sat at the end, in §4.C.
- Exercise 4.B.i(c) answer cited "(A)/(B) entries of §6.A" and "link 3 of §4.A".
- §5.A was an unsourced mini-derivation.
- §5.B listed the results as bullets plus an interpretive paragraph that gave away 5.B.i, and then repeated them in the blurb.
- §5.C was an unsourced counting essay.
- §6 was a three-part summary essay with Estimand and Width bullets.
- Technical recheck item 8 / the PAC-Bayes correction: the false "only if SGD approximately samples the posterior" sentence was still in "What carries forward".
- The closing cited a WB6 "card".
- Validator hazard, introduced by the previous rewrite and absent from the snapshot: the non-answer text contained the literal words "Exercises" ("Exercises are mostly…", "Exercises 2.A.i and 2.C.i") and "Solutions" ("the Solutions version"). `_validate` replaces "Exercises" only in the exercises variant and "Solutions" only in the solutions variant, so these lines would have made the variants compare unequal and raised an AssertionError.

## 2. Edits made

| Section | Change |
|---|---|
| Intro / route | Rubric framing removed. Route rewritten for the new §1, §3, §4, §5 and §6, and "solutions version" lower-cased. |
| §1 | Kept the heading and the `wb4-section-1` anchor. §1.A, §1.B (rubric), Exercise 1.B.i and its answer were replaced with two sentences (no-free-lunch → candidate answers in §2–5). |
| §2 | Kept. Cut the §2.A giveaway sentence. Fixed "ideal MDL is the limit of the two-part code" to "the two-part code with $L(H)=K(H)$, … uncomputable" (critic). The 2.D.ii answer now says the two-part bracket $\log_2(\Delta_\pi\sqrt J)$ is affine-invariant and changes under nonlinear $g$ (critic). "Exercises 2.A.i and 2.C.i" became "Exercise 2.A.i and Exercise 2.C.i" (validator). |
| §3 opening | The R&H core blurb moved to the top of §3. The fine-print list was trimmed to "(plus a constant) … §3.C lists what it does not say" (critic). |
| §3.B | $K(\nu)$ is defined for lower-semicomputable $\nu$ (technical item 7), together with its $O(1)$ relation to the computing-program length for computable measures. The Estimand bullet was folded into one clause after the theorem ("a bound on the mixture forecaster's cumulative online log-loss, not on any single hypothesis"). The Bayes-mixture remark was kept as a paragraph. The relative-entropy sentence now points to the subprobability definition in 3.B.i(c) (technical #5). |
| §3.C | Legg bullet → one line, with a Legg `reading()` at the top of 3.C.i. Added a definition of $K(\omega)$ for infinite sequences and of $K(p)$ (critic). |
| 3.C.ii | Deleted, with its answer. |
| §3.D | Bridge paragraph and Gibbs caution deleted. Valle-Pérez blurb: "Argues…" plus "one-sided … computable proxy … §7 tests whether SGD samples like the posterior", labelled *optional*. The CIFAR claim was corrected to "near chance" (critic; Table 1). |
| §4 | One-sentence pitch, then "4.A Readings" (Furman; Hänni–Vaintrob, with the slide 34 ~$d^k$ NNGP sentence), then "4.B What a prior buys on sparse parity" holding Exercise 4.B.i, whose number is unchanged. The chain, tags, evidence paragraph, setting bullets, theorem restatement and "Three things to notice" were deleted. The old §4.C was removed. |
| 4.B.i answer | (b) drops "(the Gibbs estimand)". (c) now points to the rows of the §6 table and to "the search problem Furman names". |
| §5 | Title is now "Hierarchical representations". The pitch is one sentence. §5.A is one sentence. §5.B is the RHM blurb first (3 sentences after the recheck), then the three model-ingredient bullets and the $P^*\simeq n_cm^L$ line. Results bullets and the interpretive paragraph deleted. The 5.B.i hint now points to §4.A. §5.C deleted. |
| §6 | One table (prior / where / $k$-sparse parity cost / correlated vs uncorrelated RHM), plus a comparison row for gradient-trained networks and the one SLT line. §6.A–6.C deleted. |
| Closing | The WB2 §7–8 bullet uses the technical-sources wording: transferring a guarantee needs a linking argument, posterior sampling is one possible link, and PAC-Bayes does not require posterior sampling. Other pointers: WB5 §2–3 (convergence), WB5 §4–5 (linear and superposed representations), WB6 described without "card". Further study is now only the mean-field paper, since Legg and R&H moved into blurbs. |

## 3. Independent feedback and dispositions

### Critic 1 (fresh read-only Opus sub-agent, full review)

| # | Finding | Disposition |
|---|---|---|
| 1 | The docstring cites a missing `reviews/wb4-editorial.md` | **Fixed** (this file). |
| 2 | 2.D.ii's two-part bullet misleads for affine maps | **Fixed**, as above. |
| 3 | "ideal MDL is the limit of the two-part code" is wrong | **Fixed**. |
| 4 | §3's opening repeats §3.C, so the R&H blurb is not at the top | **Fixed**: opening trimmed to the bound plus a pointer to §3.C. |
| 5 | Convergence belongs to WB5 §3 as well as §2; WB2 §8 compares covariance, not the whole distribution | **Fixed**: "WB5 §2–3"; "stationary covariance on a quadratic basin". |
| 6 | The §2 derivations link no original source (MacKay ITILA ch. 28, Grünwald's MDL tutorial suggested) | **Deferred to the instructor.** Memory guidance is to limit papers to ones the instructor knows, so I did not add new sources. §2.B–2.C already point into R&H §5.9 and §9. |
| 7 | The regret bound for $M$ needs "+O(1)", and the two definitions of $K(\mu)$ need reconciling | **Fixed**. |
| 8 | $K(\omega)$ is undefined for infinite sequences | **Fixed**, with the monotone machine and $K(p)$ added after Critic 2. |
| 9 | Valle-Pérez's CIFAR bound (0.485/0.518) is not "non-vacuous" in a useful sense | **Fixed**: "near chance". |
| 10 | WB2 has no NNGP/NTK material | **Fixed**: WB2 §5 removed from the table; WB5 §7 kept, reworded after Critic 2. |
| 11 | "Proves" overstates a talk whose write-up is upcoming | **Fixed**: "Shows, in talk slides with proof appendices (a full write-up is forthcoming), that…". |
| 12 | §2.C prose and §3.B state results the exercises then derive | **Kept.** These are theorem statements in derivation sections; the exercise titles already name the results. |
| 13 | WB3 cites "WB4 §2.C" (subsection level) and says WB4 asks whether networks "approximate Solomonoff" | **Not mine to edit.** Reported to the WB3 owner (see below). |

### Critic 2 (second fresh read-only Opus sub-agent, targeted recheck)

Read the full revised file and the WB2/WB5 builders, and checked the Valle-Pérez, Legg and RHM PDFs and the Hänni–Vaintrob slides.

**Passes:**

- Algebra of 2.D.ii and 3.B/3.B.i.
- The Valle-Pérez wording (Table 1: MNIST 0.134/0.169, Fashion 0.175/0.188, CIFAR 0.485/0.518).
- Hänni–Vaintrob slide references 10–12, 15–17 and 34.
- WB5 §2–3 and WB2 §8 references, and the PAC-Bayes sentence.
- §6 table syntax.
- `<details>` balance, anchors, answer-cell headings and string escapes.
- No remnants.

| Finding | Disposition |
|---|---|
| "the bracket changes" should be "in general" | **Fixed**. |
| §2.D's ideal-MDL clause reads as if it gives $\tfrac p2\log n$, and the table says MDL is computable | **Fixed**: separate sentence, "*Ideal* MDL … inherits Kolmogorov's uncomputability". |
| "computable environment" should be "computable measure" | **Fixed** in §3.B and the §3 opening. |
| 3.C.i should use the monotone machine and define $K(p)$ | **Fixed**. |
| "WB5 §4–5 = factored (compositional)" is a stretch | **Fixed**: "Linear and superposed feature representations". |
| WB5 §7 never mentions NNGP/NTK | **Fixed**: "kernel-ridge learning curves in WB5 §7". |
| RHM blurb was 4 sentences; "§III.4" is "§III.D" in the paper (Critic 1 had read "§III.4" off the HTML) | **Fixed**: 3 sentences, with the section cited as "§III … ending with the uncorrelated control", which avoids the numbering conflict. |
| Two "core" labels in §3 | **Fixed**: Legg is now "optional; core for this exercise". |
| "the SLT day" is a timetable reference | **Kept.** It is a cross-reference to another course component, not a shared schedule. It was in earlier versions too. |
| "Rules equally short" in the RHM is only approximate | **Kept.** The 5.B.i caveat bullet covers it. |
| Validator "Exercises"/"Solutions" check "couldn't fail" | **Disagree.** `_cell` substitutes only `{{VARIANT}}`. A literal "Exercises" in body text survives in the solutions variant, where `_validate` does not replace it, but becomes `{{V}}` in the exercises variant. The pre-fix text would have failed. The current file has no such literal (grep confirmed). |

## 4. Recheck outcome

Critic 2 was the targeted recheck. I made its follow-up fixes myself and then rechecked them **myself** (no third agent), by reading the changed lines and grepping for:

- "Exercises" and "Solutions"
- `\"` inside raw strings (only two remain, both in ordinary strings)
- "§III.4", and anchor and `<details>` counts (6 anchors; 9/9 details).

Both critics independently re-derived all worked answers (2.A.i–5.B.i) and found no other mathematical errors.

## 5. Remaining limitations

- **`--check` not executed** (no shell anywhere in this session). Validity is by inspection only.
- §2's derivations have no original-paper link beyond R&H section pointers (Critic 1 #6, deferred to the instructor).
- Not verified: the *Phys. Rev. X* and *Entropy* citation details; the unsourced one-line depth-separation statement in §5.A (standard, but no source linked); the content of Furman's post beyond its date and section names (Critic 1 checked those); R&H section numbers (Critic 1 checked them against the table of contents; Critic 2 did not).
- The circuit-prior slides were read directly from the PDF by me and by Critic 2. Slide 34's NNGP $\sim d^k$ is the authors' heuristic picture, and the blurb says "describes".

## 6. For other notebook owners

- **Anchors** `wb4-section-1` … `wb4-section-6` are unchanged, and §2 and §3 numbering and titles are unchanged.
- **Retitled:** §5, "Hierarchical and factored representations" → "Hierarchical representations".
- **Deleted subsections:** §1.A, §1.B, 1.B.i, 3.C.ii, the old 4.A chain, the old 4.C (readings moved to 4.A), §5.C, and §6.A–6.C (now one table). §4.B is now the sparse-parity exercise section.
- **WB3** (`build_wb3.py`): line ~184 cites "WB4 §2.C". It still exists, but a main-section citation ("WB4 §2") would be safer. Line ~535 says WB4 asks whether "networks approximate Solomonoff induction" or "search for short programs" is a defensible strengthening. WB4 no longer poses that as an explicit question (3.C.ii was deleted). It now gives the readings (§3.D Valle-Pérez, §4.A Furman) and the sparse-parity and RHM exercises.
- **WB6** (`build_wb6.py`): its text calls WB4 a "debate between competing accounts". WB4 now offers a comparison table (§6) rather than a debate framing.

## 7. Source-link completion and targeted technical recheck

The technical auditor added concise source links only in `build_wb4.py`: MacKay's original *Bayesian Interpolation* paper via its Caltech repository record for §2.A; a direct link on the existing Rathmanner–Hutter pointer for §2.B; Grünwald's MDL tutorial §2.4.2 for §2.C; and Poggio et al., Theorems 1–2, for §5.A. These satisfy the instructor's request to link source readings without expanding the discussion. MacKay's own book host returned an access error, so the accessible original-paper repository record was used instead; its abstract explicitly identifies evidence and Bayesian Occam model comparison. The PDF download could not be fetched in this tool session, so its full text was not re-audited.

Grünwald's live text explicitly discusses the coarser $O(1/\sqrt n)$ parameter precision in §2.4.2. Poggio et al.'s live text gives the shallow $O(\varepsilon^{-n/m})$ bound in Theorem 1 and the hierarchical $O((n-1)\varepsilon^{-2/m})$ bound in Theorem 2, with their smoothness and activation assumptions. The link labels point readers to those assumptions; this is an approximation-class comparison, not a lower bound for every individual compositional function.

The narrow technical recheck confirms that the semimeasure telescoping root factor, $\ln\xi(\epsilon)$ regret term, subprobability-divergence proof, definition of lower-approximation complexity, and corrected PAC-Bayes transfer statement all survived the editorial pass. No outstanding blocker remains from the previously reported WB4 findings. Python AST parsing passed. Generated notebooks were not changed by this link-completion pass; rebuilding and notebook-pair validation remain with the root agent.
