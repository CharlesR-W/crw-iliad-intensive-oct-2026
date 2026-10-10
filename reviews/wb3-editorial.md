# WB3 ("Why LLMs?") editorial pass, 2026-10-08

Owner: Claude Opus 5.5 agent. Files edited: `notebooks/build_wb3.py` only, plus this record. No `.ipynb` files, other builders, `nbkit.py`, specs or memories were touched, and no build was run. The current builder was the starting point. The pre-rewrite snapshot at `/tmp/mdl-before-20261008/notebooks/build_wb3.py` was used for comparison only. The previous owner's corrected math was kept.

## 1. Initial critique: reviews reconciled against the current builder

Governing documents: `reviews/2026-10-08-editorial-brief.md`, `reviews/reading-format-check.md` (WB3 rows) and `reviews/technical-sources.md` (WB3 items).

- **Technical items already fixed by the previous owner:**
  - #6 / recheck 6: "rather than building something new" is gone, and $\beta>0$ is stated.
  - #7 / recheck 4: the 1.1 answer now uses code-length language.
  - Fixed $T$ and "not shortest code" in §3.B.
  - Recheck 5: the interpolation qualifier is now in §5.C.
  - The persona exercise assumes identical output distributions.
- **Technical items still open:**
  - The §6 result line still stated janus/Nardo theses as results. This is the "simulator summary exceeds the derivation" paragraph.
  - The equal-mean versus equal-exponential-moment point (#6) was only implicit.
  - The interpolation qualifier lived in §5.C, which the reading-format check says to delete.
- **Reading-format items: all still open.**
  - §1.A narrated the cake, and "What the count does not show" remained.
  - §2 was a roughly 450-word essay with an unattributed weak/strong distinction.
  - Delétang was not a `reading()` beside §3.B.
  - The GDL and Vaswani blurbs sat after the §4.C prose, which stated Hooker's thesis in the §4 "**The result.**" line. The Mamba sentence was unsourced.
  - §5.A was a Kaplan→Chinchilla history, and §5.C repeated Exercise 5.3.
  - §6.A–B and §6.D were narration, with the janus, Nardo and RL's Razor blurbs placed at the end of the section. §6.D also had the selection-versus-search paragraph.
  - Exercise 6.2(c) pointed to "§6.B" narration.
- **Cross-notebook references needing change:**
  - "WB4 §2.C" should be cited at section level.
  - The closing said WB4 asks whether networks "approximate Solomonoff". WB4 no longer does.
  - "WB1 §5.D" and "WB1 Exercise 3.5(a)" needed updating, because WB1 is being revised; they are now cited at section or exercise level.
  - "WB1 §4" appeared in the janus narration.

## 2. Edits (`notebooks/build_wb3.py`)

- **Intro and route.** The opening now says what derivation parts and reading parts contain, replacing "each section opens with its main result". The route was updated:
  - §2 is "Reading only; the shortest section".
  - §4 mentions symmetry and locality, the blueprint, and readings.
  - §5 lists Exercises 5.1–5.3 accurately.
  - §6 is described as readings, two optimum exercises, then readings.
- **§1.**
  - Deleted §1.A and "What the count does not show".
  - Moved the slide quotes into Exercise 1.1(d).
  - The result line no longer gives away the 1.1 numbers and keeps the code-length caveat. The subsection is renamed "1.A. Exercise: code lengths of labels and of text".
  - Answer (d) no longer asserts that text "won".
  - The LeCun blurb is phrased as his argument ("Argues…").
- **§2.**
  - Three sentences ("Unsettled. For text… For video…") plus the LeCun blurb. The "read it as the strongest…" clause is trimmed.
  - Deleted §2.A–2.C, the weak/strong distinction and the "follow the proof" claim.
- **§3.**
  - Added a Delétang et al. `reading()` at the top of §3.B, pointing to its §2 Background on arithmetic coding. Section numbering was verified on arXiv HTML.
  - Exercise 3.2 now states the 1 GB dataset size and 2 bytes per parameter its answer uses.
  - Changed "WB4 §2.C" to "WB4 §2".
  - Derivations unchanged.
- **§4.**
  - New result line: the two exercise results, plus "symmetry together with scale separation" for the blueprint, plus "why the transformer won is a separate question".
  - 4.A is retitled "CNNs: symmetry and locality", with one clause noting that locality is a separate choice.
  - The GDL blurb moved to the top of §4.B. The post-table paragraph is now one sentence.
  - §4.C is retitled "Why the transformer: parallelism" and now holds the Vaswani blurb, a new Hooker reading (with "a general argument; it does not discuss transformers"), the complexity table and the line "WB5 §1".
  - Deleted the hardware-lottery thesis and the Mamba sentence.
  - Dropped the last clause of answer 4.2(c), "so its built-in inductive bias is weak".
- **§5.**
  - Kaplan and Hoffmann blurbs moved to the top. The Kaplan blurb follows the critic's wording.
  - Deleted §5.A (history) and §5.C. Its data-limit sentence is folded into the 5.3 answer, and its interpolation qualifier moved to the closing WB2 bullet.
  - The derivation subsection is now 5.A. The result line says the *allocation* exponents are near ½.
  - Answer 5.1(d) notes that Besiroglu et al. quote about 70 tokens per parameter for the published fit.
  - Answer 5.1(e) drops "the exponent ratio is fairly robust" and cites Besiroglu's re-fit: α≈0.35, β≈0.37, $N_{\rm opt}\propto C^{0.51}$, about 20 tokens per parameter. This was verified against arXiv 2404.10102 Table 1.
  - The 5.2(b) Hoffmann quote is now complete, verified against arXiv.
- **§6.**
  - Replaced the result line with a neutral three-sentence pitch.
  - §6.A has the janus blurb, then the persona-mixture equation, flagged as a modelling assumption. The conditioning sentence was corrected to a posterior-weighted mixture.
  - §6.B is the Nardo blurb only.
  - In §6.C, the caveats are cut to one sentence, the DPO link was added, and the support/capability scope sentence is kept.
  - Exercise 6.1 points to WB1 Exercise 3.5.
  - Exercise 6.2:
    - (b) now also asks whether equal *mean* reward suffices; the answer gives the $e$ versus $(1+e^2)/2$ counterexample from technical-sources #6.
    - (c) states Nardo's claims (cheap Waluigi, evidence asymmetry and attractor, RLHF "may" worsen) in the prompt.
    - The answer refers to "Nardo's attractor argument". The (a) hint and answer justify the joint over $(z,y)$ via Exercise 6.1 on pairs and the KL chain rule.
    - Answers speak of "the optimum", not of what RLHF does.
  - §6.D is retitled "How far does RL move the model?". It has one framing sentence, the RL's Razor blurb (scoped to tested settings, "not a result that RL cannot add capabilities") and a separate optional Mukherjee reading. That reading is scoped to the tested models and notes the bfloat16 artifact from §6 of RL's Razor, pointing to WB6 §11.
  - Deleted the KL-leash narration, the selection-versus-search paragraph and the pass@k test.
- **Closing.**
  - WB4 is described as comparing program length, circuit size and kernel norm in §3–6, with no "approximates Solomonoff" wording.
  - The WB2 bullet carries the one-epoch interpolation qualifier.
  - WB1 §3 and §5 are cited at section level, and WB6 §11 is kept.

## 3. Validation

No shell was available, so the validator was not run (`python build_wb3.py --check` was not executed). I validated by hand with grep and reading:

- All 7 `](#…)` links have matching `<a id>` anchors.
- There are no local (non-http) links and no `<details>`.
- No answer cell starts with a heading.
- The literal capitalised "Exercises" and "Solutions" appear nowhere outside `{{VARIANT}}`.
- The `markdown(...)`/`reading(...)` concatenation was traced by eye. The second critic independently traced it and found it balanced.

Python import or parse was not executed.

## 4. Independent feedback (critic 1, fresh Opus sub-agent, read-only) and dispositions

The critic found no must-fix items. It confirmed that all WB3 recommendations in `reading-format-check.md`, and the WB3 technical items, were applied. It recomputed all exercise arithmetic and found no errors.

| # | Finding | Disposition |
|---|---|---|
| 1 | §6.A "continues as the persona that best explains it" is wrong under the mixture | Fixed: posterior-weighted mixture |
| 2 | §4 "insight is symmetry" was a thesis under "The result."; locality, and GDL's scale separation, were ignored | Fixed: result line rewritten, 4.A retitled, locality clause added |
| 3 | Hooker does not discuss transformers, yet the framing implied she did | Fixed: 4.C retitled; Hooker `about` says so |
| 4 | 6.2(c) "RLHF makes" is unhedged | Fixed: "may make" |
| 5 | 5.1(e) "exponent ratio robust" is contradicted by Besiroglu's re-fit | Fixed, with the re-fit numbers verified on arXiv |
| 6 | Exercise 3.2 not solvable quantitatively from its prompt | Fixed: 1 GB and 2 bytes per parameter added |
| 7 | Route item 5 omitted 5.2 | Fixed |
| 8 | "both exponents near ½" could be confused with α, β | Fixed |
| 9 | Truncated Hoffmann quote | Fixed, verified verbatim |
| 10 | §6.D "such training" too narrow | Fixed: "RL fine-tuning" |
| 11 | RL's Razor `about` implied the paper has no theory | Fixed: "mostly empirical, with a theoretical result in an idealized setting" |
| 12 | "its own §6" ambiguous | Fixed |
| 13 | §6 intro and 6.2(a) spoke of what RLHF does, not the optimum | Fixed |
| 14 | §1 result line gave away the 1.1 answers | Fixed |
| 15 | §2 repeated LeCun's point three times and claimed "most fully" | Fixed |
| 16 | §3.A, the §4.B paragraph and the §4.C table duplicate material | §4.B paragraph cut to one sentence. §3.A kept: a short mathematical remark the earlier audit accepted. Table kept: retaining it was explicitly instructed |
| 17 | Add "(but see WB2 §5)" to §4.A's bias–variance pointer | Declined: a minor nuance, and adding cross-notebook caveats would re-grow prose |
| 18 | 6.2(a): the joint over (z,y) needed justification | Fixed in the hint and the answer |
| 19 | §3.C prequential code has no source | Declined: kept as a short bridge without adding a paper (the instructor prefers papers he knows) |

## 5. Targeted recheck (critic 2, a second fresh Opus sub-agent, read-only)

All 17 targeted items were judged resolved, and no new mathematical or factual errors were found. It independently verified:

- the Besiroglu α, β and $C^{0.51}$ figures;
- the full Hoffmann quote;
- Delétang's 1 GB and 2-bytes-per-parameter figures;
- RL's Razor §6 (bfloat16) and Theorem 5.2;
- that the Hooker essay has no transformer discussion;
- the 6.2(a) lifting argument;
- the regression checks for anchors, capitalised words, answer headings and Python string balance, and every internal and cross-workbook reference.

Its new findings and dispositions:

| # | Finding | Disposition |
|---|---|---|
| N1 | Route item 4 still framed symmetry as the thesis | Fixed |
| N2 | 6.2(b) answer said "KL-regularized RLHF cannot separate" | Fixed: "the KL-regularized optimum" |
| N3 | 6.A: "how well each explains it" ignores the prior | Fixed: "posterior probabilities" |
| N4 | "same objective" over pairs is a lifted objective | Fixed: "analogous objective", plus the KL chain-rule sentence |
| N5 | §3.C four sentences; §4.C table | Declined, as for 16 and 19 above |
| N6 | 5.1(d) 90 vs Besiroglu's quoted ~70 tokens per parameter | Fixed: parenthetical saying the ratio depends on the budget and is sensitive (no unverified cause asserted) |
| N7 | 6.2(c) answer "mode collapse sharpens attractors" is unhedged | Fixed: "may sharpen" |
| — | Unverified: whether janus "argues this side" of text versus video | Reworded to "develops this idea" |

The N1–N7 fixes were small wording edits made after critic 2's pass. I rechecked them myself with grep and reading, not with a third agent.

## 6. Limitations

- `build_wb3.py --check` was not run (no shell), and the notebooks were not regenerated. The `.ipynb` files are therefore stale until the build is run.
- The following were not re-fetched in this pass; they predate it and were checked from memory by critic 1:
  - Kaplan's α_N≈0.076 and α_D≈0.095;
  - GDL §2.2 and §3.5;
  - the janus section headings;
  - the LeCun slide quotes;
  - the LeCun OpenReview page (bot-blocked).
- Cross-references to WB1 §3/Exercise 3.5 and §5 assume WB1's ongoing revision keeps those numbers.

## 7. Anchor and title changes for other owners

- Anchors are unchanged: `wb3-section-1`…`6` and `wb3-closing`. Main section numbers and titles are unchanged.
- Subsection changes:
  - §1: old 1.A is deleted; the exercise is now "1.A. Exercise: code lengths of labels and of text".
  - §2: no subsections.
  - §4: "4.A. CNNs: symmetry and locality"; "4.C. Why the transformer: parallelism".
  - §5: old 5.A and 5.C are deleted; the derivation is now "5.A. The compute-optimal allocation".
  - §6: "6.B. The Waluigi effect"; "6.D. How far does RL move the model?".
- Exercise numbers are unchanged: 1.1, 3.1, 3.2, 4.1, 4.2, 5.1–5.3, 6.1, 6.2.
