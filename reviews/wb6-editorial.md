# WB6 editorial revision: "Mysteries myriad"

WB6 owner (Claude Opus 5.5), 2026-10-08. Governing instructions: `reviews/2026-10-08-editorial-brief.md`. Standard for narration vs blurb: `reviews/reading-format-check.md`. Source scope: `reviews/wb6-source-check.md`. Files edited: `notebooks/build_wb6.py` only (plus this record). No generated `.ipynb`, other builders, `nbkit.py`, TD materials, specs or memories were touched.

**Tooling limitation, stated plainly:** neither this session nor any sub-agent I dispatched (one Sonnet runner, two Opus critics) had a shell tool, so `python build_wb6.py --check` was **not run** by anyone in this process. The validator was simulated by hand twice (by me and by the recheck critic); see "Validation" below. The root process should run `cd notebooks && python build_wb6.py --check` (expect `stale: ...`) and then regenerate.

## 1. Initial critique of the original builder

The original had 14 "cards" plus intro. Each card had a "**The result.**" paragraph, a framing essay (about 150–300 words), one or two `reading()` calls with long "about" text (read-this-section-then-that guidance), several with `note=` aimed at CRW, three questions with answer cells, and a `commentary()` call naming CRW's view. The intro had "45 minutes per card" and a "before 17:00" convergence instruction.

Classification by block type:

| Block | Classification | Action |
|---|---|---|
| Intro: fact list, "Time", "How to use", convergence-card instruction | Narration + schedule | Replaced by 2-sentence description, route list, variant statement |
| "**The result.**" paragraphs (§1–§14) | Narration restating the paper | Deleted; the content that matters went into the 1–3 sentence blurb |
| Framing paragraphs (§1–§14) | Narration (incl. "CRW's note for this card reads…", §11, §14) | Deleted |
| `reading(... about=...)` guidance essays | Reading-guidance narration | Deleted; blurbs now `result` only |
| `note=` and `commentary(...)` (CRW-aimed) | Instructor-directed / attributed views | Deleted; `commentary` import removed |
| Q1.1–1.3, 2.1–2.3, 3.1–3.3, 4.1–4.3, 5.1–5.3, 6.1–6.3, 7.2–7.3, 8.1–8.3, 9.3, 10.2, 10.3, 12.1–12.3, 13.1–13.3, 14.1–14.2 | Generic (mechanism / what-would-falsify / design-an-experiment / which estimand / "a good answer…") | Deleted. Q3.2 was also a predict-before-reading gate. Q4.3/7.2/10.3 used the Bayes/Gibbs estimand theme the brief rejects as unifying. |
| Q7.1 (bits per label), Q6.2 (generalization measure with rescaling) | Calculation-flavoured but answers were loose/generic | Deleted |
| Q9.2 (K outliers via Gauss–Newton) | Heuristic sketch, not a complete derivation | Deleted |
| Q9.1 (quadratic model) | Real derivation, but answer incomplete (claim that top components "keep dominating" holds only in a narrow step-size window) | Kept and rebuilt as Exercise 9.1 with a complete worked answer |
| Q10.1 (BN scale invariance) | Real derivation, but answer contained an error: "the Hessian has a zero eigenvalue in the radial direction" holds only at stationary points (in general $Hw=-g$, so only $w^\top Hw=0$); outlier-suppression part speculative | Kept, corrected, speculative part removed (Exercise 10.1) |
| Q11.1 (sparse vs low-rank) | Real but trivial; answer had an unsourced "LoRA works well for RL" claim | Kept, strengthened with a generic-rank proof (Exercise 11.1), unsourced claim removed |
| Q11.2 (bf16 rounding) | Real calculation; rested on an unverified description of the paper's measurement | Kept after verifying the measurement in the paper's HTML (abs. diff ≤ 1e-5 on bf16 values); rebuilt as Exercise 11.2 |
| §14 "What kinds of explanations do we need?" | No reading; generic rubric (14.1–14.2), "what carries forward", further study, convergence card | Deleted entirely, including the card template and its answer cell |

## 2. Edits (final structure)

- **Intro:** title, variant line, two sentences on what the notebook is, a plain statement that the exercises and solutions variants are identical apart from the variant label and the four worked answers (§9–§11), a 13-line route list (`[§k. Title](#wb6-section-k): one-line pitch`), and one line pointing to Training Dynamics.
- **§1–§13:** heading with anchor `wb6-section-k` (titles unchanged), one `*Connects to:*` line at main-section level, and 1–3 `reading()` blurbs of 1–3 sentences.
  - Connects-to lines: §1 WB2 §6, WB5 §3; §2–§3 WB5 §2; §4 → §8; §5 WB5 §5; §6 WB2 §9; §7 → §8 (pruning); §8 WB5 §2; §9 WB2 §8; §10 WB2 §4, §9; §11 WB3 §6; §12 WB4 §4; §13 WB4 §4, WB5 §4–§5.
- **Readings:** all 16 original readings kept (breadth preserved). One added: Cai et al. 2025 (arXiv 2510.00553, "rank-1 dominance") in §11, already cited in `td/build_td2_8_further_readings.py`; added so the section titled "Low-rank RL updates" has a low-rank-side reading. Abstract checked.
- **Exercises (4 answer cells):** 9.1 quadratic toy (overlap vs where loss decreases; step-size window $2/(\Lambda+\mu)<\eta<2/\Lambda$; numerical example); 10.1 scale invariance under BN ($w^\top g=0$, $\alpha^{-1}$/$\alpha^{-2}$ scaling, $Hw=-g$, norm growth and effective step $\eta/\lVert w\rVert^2$); 11.1 sparse vs low-rank (generic full rank of a support containing a permutation pattern); 11.2 what "unchanged" means under the bf16 + 1e-5 measurement.
- **Module docstring** rewritten to describe this structure; `commentary` import removed.

Source checks performed by me for blurbs not covered by `wb6-source-check.md`: arXiv abstracts of 2502.02013 (32 tasks; architectures), 2505.11711 (7 algorithms, 10 LLMs, "conjecture", near-full rank), 2505.11711 HTML (1e-5 bf16 tolerance; SFT 6–15% sparsity), 2509.04259 HTML §6 (bf16-caused sparsity, float32 control, full-rank updates), 2510.00553 abstract, and CRW's essay (part structure, co-selection).

## 3. Independent feedback and dispositions

**Critic 1** (fresh Opus sub-agent, read-only, not told my conclusions). Sentence counts: all blurbs 1–3; no narration found; no leftovers; all four exercises worked independently and found correct, 11.2(d) incomplete.

| # | Finding | Disposition |
|---|---|---|
| 1 (high) | §11 omits RL's Razor §6: in their setting sparsity came from bf16 and vanished in float32, full-rank updates | **Fixed.** Verified in arXiv HTML; added to RL's Razor blurb, 11.2(d) answer; 11.1(c) hedged; §11 route line now "reported as sparse…" |
| 2 (med) | Title "Low-rank RL updates…" but no low-rank reading | **Fixed** by adding Cai et al. (already cited in TD). Title kept: WB3 cites it verbatim and TD links the anchor |
| 3 (med) | Song, Ahn & Yun 2025 (arXiv 2405.16002) tests whether SGD happens in the subspace; suggests adding | **Not added**: not cited in any notebook builder (only in `CURRICULUM_DRAFT_v1.md`, marked unchecked), so outside the brief's rule for additions. Answer's last sentence reworded so it does not claim the question is untested. Flag for instructor |
| 4 (low–med) | Exercise 10.1 has no source | **Partly**: setup now says "This calculation is not in the paper". No source link added because the candidates (van Laarhoven 2017; Arora et al.) are not cited in any notebook. Flag for instructor |
| 5 | §13 connects-to names no WB5 section | **Fixed** (WB5 §4–§5) |
| 6 | §7 has no pruning reading | **Fixed** in route line ("pruning is in §8"); title kept |
| 7 | Hinton blurb teases left-out-digit result | **Fixed** (says what it shows; see recheck item) |
| 8 | 11.1(c) relies on unstated matching threshold | **Fixed** (hint states it) |
| 9 | 9.1(c) assumes nonzero initial coefficients; boundary case | **Fixed** |
| 10 | 11.2(b) power-of-two edge | **Fixed** |
| 11 | Mukherjee "nearly identical model" | **Kept**: this is the abstract's own wording |
| 12 | Ilyas "imperceptible" vs "incomprehensible" | **Fixed** ("incomprehensible", abstract wording) |
| 13 | Bare URL in §12 `who` | **Fixed** (markdown link) |
| 14 | Intro variant statement omits title-line difference | **Fixed** |
| 15 | Docstring cites this file before it existed | **Fixed** (this file) |
| 16 | Other notebooks still say "WB6 card(s)": `build_wb5.py` lines ~118, 347, 535, 544; TD `build_td2_8_further_readings.py` ~615 ("WB6 card 11"); 11.2 overlaps TD 2.8 Ex. 5.1.1 | **Not editable by me**; reported to root. Links/anchors still resolve. Overlap with TD accepted |

**Own fix during this round:** the intro originally said "The Exercises and Solutions variants…". `nbkit._validate` compares `ex.replace("Exercises","{{V}}")` with `so.replace("Solutions","{{V}}")`, so a capitalized "Exercises"/"Solutions" in prose makes the two sides differ and would raise an AssertionError. Rewritten in lowercase (as WB3 does). The recheck critic asserted this check "could never fail"; I disagree, by the replacement logic just described, and kept the lowercase wording.

**`reviews/wb6-math-check.md`** (a concurrent read-only technical audit by another process, found by the recheck critic) was also applied: BN-invariance assumptions (batch statistics, ε = 0, nonzero variance, no weight penalty, twice differentiable); 10.1(d) no longer claims curvature monotonically shrinks ($H(w_t)=\lVert w_t\rVert^{-2}H(u_t)$, and $H(u_t)$ moves); 11.1 "rank at most $r$"; 11.1(c) conclusion made conditional on generic/independent supports; 11.2 normal-number scope, exact-representable start, $|w|>2^{-9}$, $0.02\to0.02001953125$, "0.3%" limited to the example. Its suggested van Laarhoven / Ioffe–Szegedy links were not added (not cited in notebooks).

## 4. Targeted recheck

**Critic 2** (second fresh Opus sub-agent, read-only). All seven targeted fixes FIXED or mostly fixed; blurb counts 1–3 (17 blurbs); no narration or leftovers; 9.1(d) and 11.2(c) arithmetic re-verified (and matched `wb6-math-check.md`). Residual items and dispositions:

- Cai "about 99%" → abstract says "over 99%": **fixed**.
- Hinton: student already gets ~87% of test 3s right before the bias shift (98.6% after); speech result distils an ensemble into a same-size model, so "small model" was wrong: **fixed** (blurb now distinguishes MNIST larger-model teacher from speech ensemble teacher, and gives 98.6%).
- 10.1(d) overclaim and 11.x scope items from the math check: **fixed** as above.

After critic 2, I made only these wording and scope edits and re-read the changed lines myself. No third critic was run.

## 5. Validation

`--check` was not executed (no shell available; see top). Hand simulation of `nbkit._validate` (by critic 2 before the final edits, and by me after them): Python syntax balanced; no capitalized "Exercises"/"Solutions" in any cell; all `](#wb6-section-k)` links (route list and the two §8 links) have matching `<a id>` tags; no local link targets (all links are https or `#`); no answer cell starts with a heading; no `<details>`. The expected result is `stale: 06_mysteries_myriad_exercises.ipynb, 06_mysteries_myriad_solutions.ipynb`, but this has not been confirmed.

## 6. Limitations and items for the root/instructor

- `--check` not run; generated notebooks not regenerated.
- Not checked at full-paper level: blurbs for SVCCA, stitching, critical periods, Zhang et al., Ilyas, Elsayed, lottery tickets, LMC, Gur-Ari, Ghorbani and Allen-Zhu & Li rely on `wb6-source-check.md` (abstract-level), the original builder and critic 1/2's spot checks. Specific details not independently verified by me: "under 10–20%" and the warmup detail (Frankle & Carbin), "ResNet-50 on ImageNet" (LMC abstract wording recalled by me, not re-fetched), Elsayed's "two classes" forced choice, and the venue strings in `who` fields.
- Candidate additions for the instructor (not added, per the "already cited" rule): Song, Ahn & Yun 2025 (§9); a source for Exercise 10.1 (van Laarhoven 2017, arXiv 1706.05350).
- §11's title "Low-rank RL updates and the KL leash" kept for cross-reference stability, even though its readings disagree about rank. Renaming it would require edits to WB3 (lines ~501, 538) and the TD link text.
- The original task brief said WB3 cites "WB6 §1". In the current WB3 I found only "WB6 §11" references (lines ~501, 538). §1 kept its topic anyway.

## 7. Anchor and title changes other notebooks need to know about

None. Anchors `wb6-section-1`…`wb6-section-13` and all 13 titles are unchanged. Removed: §14 and anchor `wb6-section-14`. No other file referenced `wb6-section-14` (grep). The "card" terminology is gone, so `build_wb5.py`'s and TD's "WB6 card(s)" wording should become section references.

## Root integration after owner completion

Removed §13 (the agent-added CRW essay) and its route entry, following the stated default interpretation of the user's “drop my views; at most I'll add them.” No incoming Mysteries-workbook links target that section. Sections 1–12 retain their anchors. A focused mathematical recheck is recorded in `wb6-math-check.md`; its residual 11.2(d) threshold wording was corrected. The builder was executed successfully; final freshness checks are recorded separately.
