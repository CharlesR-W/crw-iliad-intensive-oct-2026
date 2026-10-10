# WB2 design/organization revision — 2026-10-08

Reviser: Claude Opus 5.5 (high effort). Scope: `notebooks/wb2_cells_a.py` and `notebooks/wb2_cells_b.py` only. I did not edit `nbkit.py`, the README, other builders, generated notebooks or historical specs. I did not regenerate or run the checks; that is for root. Authority: `reviews/2026-10-08-editorial-brief.md` and the user's direct instructions for this task. Inputs: `reviews/wb2-technical.md` (math fixes, which I preserved) and `reviews/wb2-editorial.md` (Opus editorial review, mostly adopted).

## 1. Critique of the pre-revision source

- **Opening.** It had a "Main results" list duplicating each section's own result paragraph, a separate "Route" list, and a "Time" block (3–5 hours, "Read §1 first (about 30 minutes)"). It claimed the sections were independent, which is false for §5, §8 and §9. It told students to "commit before you open" figures. A "Three estimands" block made Bayes/Gibbs/MLE a notebook-wide rubric ("every section says which one it means"). Solutions were mentioned but not linked.
- **Attributed views and placeholders.** These were the "CRW's position" paragraph (§6.A), "closest to CRW's position" (§6.A and the §6.F Goldfeld reading), and the empty "CRW's Morse–Bott note" (§8.D). Eight `commentary(...)` calls carried attributed prompts, and several `note=` arguments mentioned CRW.
- **Gates and schedules.** There were prediction gates at §4.C, §5.C, §8.C.iii and §9.B, plus the title page. The closing cell had "What carries forward", "Further study" and a "Convergence card" ("Bring one card to the 17:00 convergence") with an example answer.
- **Generic discussion.** §6.A.i asked students to "commit now" to discriminating evidence before the derivations, with a paired "has your answer changed?" at §6.F.i. §2.D.i.3 was a write-the-reply exercise. §5.D.i opened with "which parts would carry over". There was a padding paragraph after §2.A, and 2.C.i.4 duplicated 2.D.i.1.
- **§9.** It contained essay-length setup prose, three reading questions, a 9.B exercise built on a "supplied sweep table (to be added)" that does not exist (its answer was marked "*Pending*"), and a 9.C "Discussion".
- **References.** Several derivations had no source pointer: NFL, the Occam bound, the depth separation, ridge/df, the Hoerl–Kennard and Wager–Wang–Liang mentions without links, early stopping, optimal-ridge monotonicity, IB self-consistency, the sandwich/AIC/BIC, and the SGD stationary law.
- **Hints.** None for 4.B.i, 4.C.i, 7.B.ii or 8.C.ii.
- **Minor errors.** The §8 result paragraph had a garbled sentence ("On a quadratic basin,\nIn the continuous-time approximation"). The 6.F.i answer said noise-free $I(X;T)$ "cannot change", contradicting the technically corrected 6.B (it is constant only under injectivity). The Zhang–Yang–Tu reading said commutation follows ("so") from $C\propto\mathbb E[h^2]$, which the technical review rejects.

## 2. Revision made

**Opening (part A).**
- Replaced "Main results", "Route", "Time" and "How to use" with:
  - a **Worked solutions** paragraph that links both `02_statistical_learning_solutions.ipynb` and `02_statistical_learning_exercises.ipynb` and says both are for everyone;
  - a **Sections** table (link, what you derive or read, "Uses"). The "Uses" column gives honest prerequisites (e.g. §5 uses §1.B.ii, §4.A.i.1, §4.C.i.4; §8 restates §7.A–7.C). There is no order, time or completion expectation.
- Deleted the "Three estimands" block. The opening now promises source links per section. The notation block is unchanged.

**§1.**
- Added an ESL §3.2/§7.4 pointer.
- Trimmed the estimand phrasing in the 1.A.i.4 answer.
- Removed `commentary`.

**§2.**
- Added sources: Wolpert (1996) NFL; UML §5.1, §2.3.1, §7.3, ch. 4, ch. 6; Leshno et al. (1993); Telgarsky (2015) for the depth separation.
- Rephrased 2.A.i.4 from "a prior you think is closer to the world's" to a concrete question.
- Deleted the post-NFL padding cell.
- Deleted 2.C.i.4 and its answer (duplicate of 2.D.i.1).
- Split the hard-to-read VC sentence in the 2.C.i.2 answer without changing its content.
- Deleted 2.D.i.3. Its concrete predictions of a soft inductive bias moved into the 2.D.i.1 answer.

**§3.**
- Reworded the result paragraph and 3.A.i.3 so the ensemble/posterior-draw identity no longer depends on the deleted opening rubric. §7.E is named as the place the terms are defined.
- Added ESL §7.3/§3.4.1 and a Hoerl–Kennard (1970) DOI link.
- Removed `commentary`.

**§4.**
- Added hints to 4.B.i and 4.C.i. The 4.B.i hint points to UML §26.2 for the Rademacher complexity of linear classes.
- Added a pointer to Ali, Kolter and Tibshirani (2019) for gradient flow versus ridge.
- Deleted the "Commit before you look" gate; the figure stays.
- Linked Wager, Wang and Liang (2013) for dropout.
- Removed `note=` and `commentary`.

**§5.**
- Added Belkin et al. (2019) for the term double descent, and Nakkiran, Venkat, Kakade and Ma (2020) as the source for 5.C.i.3.
- Replaced the sketch gate with exercise 5.C.ii (locate the minimum and the crossing at SNR 5 and 0.5, then sketch). The answer and figure are unchanged.
- Linked Chinchilla and Nakkiran et al. (2019) in 5.D.
- Narrowed 5.D.i to the concrete measurement question (ensemble variance decomposition via §3.A), dropping the speculative "what carries over" half.

**§6.**
- Linked Tishby–Pereira–Bialek. Removed "CRW's position", its commentary, and "closest to CRW's position".
- Deleted 6.A.i ("commit now") and its answer.
- Rewrote 6.F.i as a single post-derivation question: what do 6.B–6.D establish about the three published positions, and what does none of them address? Its answer keeps the specific content of the old 6.A.i answer: the invariance test, the clustering-blocking intervention, and Saxe et al.'s ReLU/full-batch observations.
- Fixed the 6.F.i answer so it matches the corrected 6.B (constant under injectivity, changes only through exact collisions, infinite if atomless).
- Removed the redundant qualifier in the 6.D.ii.3 answer. Removed the ghost reference to "the question 6.A left open" in 6.D.ii.4.
- Added Cover and Thomas ch. 8–10 pointers.

**§7.**
- The result paragraph now labels the single fit / Bayes ensemble / posterior draw comparison as the optional 7.E, and restricts the RLCT statement to the Bayes error.
- Added a sources line: Huber; White (1982); Akaike (1974); Schwarz (1978, DOI link); ESL §7.4–7.5, §7.7.
- Added a hint to 7.B.ii. Replaced "the SLT day" with "singular learning theory".
- Marked 7.E optional and moved Watanabe (2018) there, plus a Watanabe (2010, JMLR) link.
- Kept the old 7.F question (specific, with a technical-team correction about second moments) but moved it inside the optional 7.E as 7.E.ii. Removed "Discussion" and its `commentary` (which mentioned alignment).

**§8.**
- Fixed the garbled result sentence. Added Mandt, Hoffman and Blei (2017) as the source for the stationary-law analysis.
- Added a hint to 8.C.ii.
- Replaced "Commit, then look" with exercise 8.C.iii (compute the axes and aspect ratios from 8.C.i.2). The figure now comes after the solution.
- "Training Dynamics day" became "Training Dynamics materials".
- Deleted the Morse–Bott placeholder and notes.
- The Zhang–Yang–Tu reading now states commutation as a separately reported property.

**§9.** Reduced to:
- one line;
- the Jiang et al. reading, re-described from the paper itself (2187 CIFAR-10 models; over 10,000 with repeats and SVHN; 7 hyperparameters; cross-entropy 0.01; spectral-norm product strongly negative; sharpness best; gradient noise and speed predictive);
- one exercise, 9.A, with a hint. Part 1 derives that TIC is reparametrization-invariant and classifies the raw measures, reusing the checked answer. Part 2 asks why training to equal cross-entropy matters and which TIC assumptions fail; the answer reuses the technically corrected interpolation bullets from old 9.C.

Deleted the nonexistent-sweep exercise (9.B), the pooled-versus-Ψ question and the 9.C "Discussion". I dropped the earlier, unverified claim that gradient variance was "positively correlated along every hyperparameter axis".

**Closing.** Deleted "What carries forward", "Further study" (moved into §6/§7) and the convergence card with its example.

**Departures from `reviews/wb2-editorial.md`.**
- R1: the route is a table with a "Uses" column, not "Start with §1". The solutions notebook is linked directly.
- R5: one merged question, not two parts. Part 2 of R5 was a pairwise "what would discriminate" falsification prompt; its concrete tests survive inside the answer.
- R7/R10: §9 was cut further than R7 proposed, because the user asked for §9 to be a short reading description. 9.C's content survives in 9.A.2.
- 5.D.i was narrowed, where R12/"keep" listed it unchanged.
- 2.D.i.3 was removed.
- 7.F was kept but moved into the optional 7.E.

## 3. Source verification performed (this session)

- Abstract pages fetched: Telgarsky 2015 (arXiv 1509.08101), Wager–Wang–Liang 2013, Ali–Kolter–Tibshirani 2019, Mandt–Hoffman–Blei 2017, Nakkiran et al. 2020 (optimal regularization), Nakkiran et al. 2019 (deep double descent), Belkin et al. 2019, Chinchilla, Tishby–Pereira–Bialek, Watanabe 2010 (JMLR), Schwarz 1978 (Project Euclid), and the Hänni–Vaintrob LessWrong post (exists, dated August 2026).
- Jiang et al. 2019: pages 1–9 of the PDF were read. They confirm the reading description above.
- Table-of-contents PDFs were read for ESL (2nd ed.) chapters 3 and 7 and for UML (all chapters cited).
- DOIs for Wolpert 1996, Leshno et al. 1993 and Hoerl–Kennard 1970 resolve to the expected publisher pages (Neural Computation 8(7):1341; Elsevier pii S0893608005801315; Technometrics). The article pages were not opened.
- White 1982 (JSTOR) and Akaike 1974 (IEEE) pages did not load. They are cited as plain text without links. Huber 1967, Cybenko 1989, Hornik 1991 and Cover–Thomas are cited from knowledge, without links.
- I did not check page or section locators inside papers beyond those listed, and I did not re-derive math already checked in `wb2-technical.md`.

## 4. Independent critique, fixes, recheck

**Critique.** A fresh read-only Opus 5.5 sub-agent reviewed the revised sources. It was given the brief, the user's requirements and the two earlier reviews, and it was told the earlier editorial review was advisory. It found no remaining attributed views, gates, schedule language, notebook-wide estimand framing or ghost references. All cross-references resolved, and it saw no damage to the corrected math. It spot-checked Zhang–Yang–Tu, Hastie et al., Jiang et al., Leshno et al. and Hoerl–Kennard against the papers. Its findings and my fixes:

- **M1.** The opening said §7–§9 restate what they need, but §9 does not. Fixed.
- **M2.** The "Uses" column missed some prerequisites: §2→§4 (2.D.i.2); §4→§1.A.i.3/§2.C–2.D; §5→§3.A (5.D.i); §7→§1.B and §3–§4 (7.E.ii). Fixed.
- **M3.** The 6.B.i.2 question asked "how likely is a collision", but the corrected answer no longer estimates a probability. The question was reworded to match the answer.
- **M4.** Added hints to 6.D.i and 8.B.i.
- **M5.** Added sources for §4.A (ESL §3.4.1) and §8.A (Zhang–Yang–Tu §2.1, a section title the reviewer confirmed).
- **S1.** Restored the pooled-τ-versus-Ψ question as 9.A.3, with its original answer. It is short, specific, and the reading blurb already mentioned Ψ.
- **S2.** The §8 result paragraph and the 8.D blurb presented [1,2] as the measured γ range. Both now give the fitted ≈1–1.5 values, with [1,2] as the moment-argument range.
- **S3.** The 8.C.ii.2 answer referred to a figure the reader had not reached. It now names the figure after 8.C.iii.
- **S4.** The 6.F reading blurbs repeated 6.A. They are now one sentence each, referring back to 6.A.
- **Optional items taken:** removed the duplicate Wager–Wang–Liang attribution in the 4.D answer; "figure after the solution" became "figure below" (5.C.ii, 8.C.iii); clearer wording for "left and up" in 6.D.ii.3.
- **Optional items not taken:** links for Akaike, Huber, Cybenko and Hornik (unverified DOIs); a TIC citation; a WB6 pointer; a second §9 exercise; hints for 3.A.i, 4.A.i and 8.A.i (moderate difficulty, already well set up).

**Recheck.** A second fresh read-only Opus sub-agent confirmed all nine fixes. It read both CELLS lists in full for syntax and found no unclosed calls, missing `+` or commas, or broken quotes; `<details>` tags balance (13 each). It re-scanned for banned items and found none, and every cross-reference resolved. It found five further items, all now fixed:
- The §8 result paragraph still said the moment argument "confines" γ to [1,2], contradicting 8.B.i.4. It now says the argument makes [1,2] the natural range without forcing fitted exponents into it.
- 8.B still wrote "with $1\le\gamma\le2$" before the fitted values. That phrase was removed; the bound remains attributed to their §5.
- The opening now says §9 points back to §4, §7 and §8.
- The §4 and §8 "Uses" entries were made exact.
- The Saxe and Goldfeld blurbs were trimmed further.

None of these changed derivations or numerical results. The 8.B wording edit touches text written by the technical team, but only to separate the bound from the measured range, which 8.B.i.4 already does.

## 5. Remaining limitations

- I did not run `build_wb2.py --check`, `_validate` or `checks/wb2a_check.py`/`wb2b_check.py` (no shell in this task). Validation constraints were checked by reading, twice. Root should regenerate and run both checks.
- Citation verification was limited to what §3 and the reviewers' spot-checks list. White, Akaike, Huber, Cybenko, Hornik and Cover–Thomas chapter numbers come from knowledge. The Hänni–Vaintrob post exists but was not read beyond its summary. Section locators inside readings were mostly not rechecked.
- WB2 section numbers and anchors are unchanged. Other notebooks that cite "the three estimands" (WB6) now land on the optional §7.E. "Training Dynamics materials" and "singular learning theory" replace "day" wording in WB2 only; a cross-notebook decision is root's.
- `notebooks/README.md` still calls `*_solutions.ipynb` the "instructor notebook" (editorial review Q5). The current `nbkit.py` docstring already says both variants are for students. The README is outside my scope.
