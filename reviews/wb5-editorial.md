# WB5 (Universality): editorial revision record, 2026-10-08

Owner pass on `notebooks/build_wb5.py`, following `reviews/2026-10-08-editorial-brief.md`. Two other documents shaped it: `reviews/reading-format-check.md` was the worked example of the standard, and items 1–4 and 8 of `reviews/technical-sources.md` set the technical fixes. Only `build_wb5.py` and this file were edited. No `.ipynb` was touched.

## 1. Initial critique of the original builder

I classified each block as setup (S), reading blurb (B), derivation/exercise (D) or narration/padding (N).

| Block | Class | Problem |
|---|---|---|
| Intro: "Main results" list, Time, "commit to a prediction" | N | Schedule-like time budget; prediction gate; the PRH bullet overstated scope ("SGD's noise selects a single representation"). |
| §1 (≈400 words, no reading) | N | Unsourced essay; falsification question (Q1); commentary placeholder. |
| §2.A–C (≈600 words) | N/S | Exposition of CKA/CCA groups, stitching and permutation symmetry. Only the invariance groups and the GL symmetry are used later, and they are better learned as exercises. The CKA=1 statement omitted centering; "orthonormal columns" ignored unequal widths. Commentary placeholder; generic questions. |
| §3.A (Huh narration) | N | Restates the paper; the blurb came after it. |
| §3.B–E | S/D, written as narration | The real derivation, but with no exercises, and with errors: reversed inverses (audit item 1); loss $\|f-y\|^2$ inconsistent with $c$ (item 2); no $\Sigma_x\succ0$; "unique entropy-minimizing point on each orbit" (item 3). |
| §3.F, closing paragraph | N | Narrates the paper's six failure modes and its sharpening link. The scope caveat ("not a proof that SGD converges") appeared only here, after the "result" and the blurb had said networks "end up" aligned (item 4). |
| §3 discussion Q1–Q4, `note=`, commentary cell | N | Q4 is the falsification rubric; `note=` was aimed at the instructor; Q1 and Q3 were specific and worth converting. |
| §4.A–C (≈500 words) | N | Narrates Park et al. and Engels et al. The one usable derivation, the non-identifiability of the inner product, was not an exercise. |
| §5.A–C | S/D | A real derivation, gated by "Commit to a prediction" before the figure. |
| §5.D | N | Unsourced narration on adversarial examples and error correction, with a "what would decide it" question. |
| §6.A–D (≈1,100 words) | N | Narrates four papers. The one derivation (Cloud et al.'s first-order argument) was prose, not an exercise. The LoRA-artifact caveat was correct. |
| §7.A–B | S/D, with problems | Real derivation, but: a prediction gate; "exact curve" in the figure text; no boundary log factor (item 8); a Bayes/Gibbs estimand line; "CRW's … makes this argument"; `note=`; discussion questions. |
| Closing: "What carries forward", "Convergence card" | N | Card with the 17:00 session; speculative carry-forward claims; an unlinked "Ziyin et al. (2025)". |

## 2. Edits made

The whole builder was rewritten. `split_discussions` and the `commentary` import were removed, because no discussion blocks remain.

**Sections kept and cut**

- **Intro.** Now a two-sentence pitch, the route with one line per section, and "How to use" (solutions open to all; meaning of *core*). Removed: main-results list, time budget, prediction instruction.
- **§1** (anchor kept, title unchanged). Two sentences plus a Kaplan et al. blurb (shape-independence, LSTM context). The essay, both questions and the commentary are cut.
- **§2** (title unchanged). One-sentence pitch; Kornblith (core) and Bansal (optional) blurbs at the top. The minimal setup (centered $X,Y$, CKA, CCA as principal angles, mutual kNN in one line) is followed by two exercises:
  - **Exercise 2.1** derives CKA's group, CKA=1 ⇔ proportional *centered* Grams ⇔ an isometry between occupied subspaces (plus translation), and the CCA degeneracy when the rank is $n-1$.
  - **Exercise 2.2** covers permutation and ReLU rescaling, and shows that a GL insertion preserves the Gram up to scale iff $T$ is scaled orthogonal.

  §2.B (stitching) is folded into the Bansal blurb; §2.C becomes Exercise 2.2.
- **§3** (title shortened from "…(worked paper: Ziyin & Chuang)" to "The Platonic Representation Hypothesis"). A neutral pitch states the scope limit first, then the Huh and Ziyin–Chuang blurbs (both core). Setup §3.A is followed by four exercises:
  - **3.1** global minima and why being one is not enough;
  - **3.2** the balance condition from the symmetry moves, including non-uniqueness;
  - **3.3** factorization, whitening, the full solution of the two matrix equations for any width $\ge r$, the representation, and label transforms;
  - **3.4** no noise, gradient flow, weight decay (with a worked counterexample), and which of Huh's pressures is used.

  Huh narration, §3.F and the discussion questions are cut. Former Q1 becomes 3.4(d); Q3 becomes a remark in the 3.4(c) answer (MAP = weight decay). Q2 and Q4 are deleted.
- **§4** (title unchanged). Two-sentence pitch, the Park (core) and Engels (counterpoint) blurbs, then **Exercise 4.1** (logit invariance under $GL(d)$; the covariance-whitened inner product is invariant). The §4.A–C narration and the questions are cut.
- **§5** (title unchanged). The pitch keeps the interference account of adversarial examples, because WB6 §5 cites it, and says plainly that there is no reading on error correction. Elhage blurb (core). **Exercise 5.1** covers near-orthogonal packing; **Exercise 5.2** is the $n=2$, $m=1$ boundary. The figure link comes after the exercise, with no prediction gate. §5.D narration and the questions are cut.
- **§6** (title unchanged). Subsections 6.A–6.C each open with their blurbs:
  - **6.A**: Golowich, then **Exercise 6.1** (softmax rank ≤ $d+1$; why it does not bound the multi-token matrix).
  - **6.B**: Cloud, then the Nief blurb marked *contested*, then **Exercise 6.2** (the shared-initialization first-order argument, plus where it fails without a shared init).
  - **6.C**: Aden-Ali, then **Exercise 6.3** (LLS score $=u^\top\delta_i$; first DPO step; a student embedding $B\phi$).

  §6.D's "tension" narration is reduced to one clause in the Aden-Ali blurb. The questions are cut.
- **§7** (title unchanged). Pitch, then blurbs: CRW's post (neutral "Argues…", no `note=`; after critique labelled *optional* and undated), plus two new source links, Michaud et al. (quanta, optional) and Li, Zhang & Lin (rigorous KRR curves, rabbit hole). Setup, then **Exercise 7.1** (tail sums, heuristic exponents, quanta) and **Exercise 7.2** (saturation, boundary log factor, noisy exponent), then the figure. The prediction gate, the Bayes/Gibbs estimand line and the discussion questions are cut.
- **Closing.** "What carries forward" and the Convergence card are replaced by four pointers: Neural Thermodynamics (now linked: arXiv 2505.12387), Training Dynamics, WB6 §3 and §5, and WB3 §5. The WB2 §8 pointer now appears only in the Exercise 3.4(c) answer.

**Technical fixes (location → fix)**

1. §3 Exercise 3.1(a) and its answer: $W_D\cdots W_1=(M^O)^{-1}V^*Z_A^{-1}(M^I)^{-1}$, derived with explicit left/right multiplication (audit item 1).
2. §3.A: $\Sigma_x\succ0$ is assumed, with the restrict-to-range/pseudoinverse alternative. Exercise 3.1(a) asks where it is used (item 1/3).
3. §3.A and Exercise 3.3(a): $\ell=\tfrac12\|f-y\|^2$ throughout, with minimum $\tfrac12\mathrm{Tr}\,\Sigma_\epsilon$, $g=-\bar M^\top\epsilon$ with no factor 2, and $c=\mathbb E\|g\|^2/\mathbb E\|h\|^2$. The answer notes the factor 4 that $\|f-y\|^2$ would give (item 2).
4. Exercise 3.2(c)–(d): uniqueness is replaced by the stationarity implication actually used. The exercise shows convexity along lines and also that $S$ is constant along orthogonal moves, so the minimizer is not unique, and states that $(*)$ is used only as a necessary condition (item 3).
5. §3.A and Exercise 2.1(c)–(d): CKA=1 needs centered Grams; the PRH representations are linear with no bias, so centering is harmless. Translation freedom is stated (item 3).
6. Exercise 2.1(c), Exercise 3.3(c)–(d): unequal widths are handled by $R\in\mathbb R^{d\times r}$ with $R^\top R=I$, and $R_BR_A^\top$ is an isometry between occupied subspaces. The solution of 3.3(c) is proved for all widths $\ge r$, consistent with the paper's Theorem 1 ("width … no smaller than the rank of $V^*$") (item 3).
7. Scope (item 4): the §3 pitch (first substantive statement), the Ziyin–Chuang blurb, §3.A "Selection" and the 3.3(d) answer all say the theorem characterizes minimizers of the small-$\eta$ modified objective, and that SGD convergence is not proved (the paper's Remark). No "end up" wording remains. The intro was also rephrased to match.
8. Exercise 7.2(b)–(c): the boundary $\alpha'-1=2\alpha$ gives $n^{-2\alpha}\log n$ (noiseless) and an extra $(\log n)^{1/(s+1)}$ (noisy); otherwise the rate holds "away from the boundary". Deterministic equivalents are called "asymptotic approximations rather than the exact finite-$n$ risk" in the 7.2 answer and in the panel-B figure text, which no longer says "exact curve" (item 8).
9. Subliminal learning: the Nief blurb (*contested*) quotes the listing's own caveat and says it "does not settle the question either way". The about line notes that the MNIST demonstration uses no LoRA. Nothing presents it as a refutation.

**Source checks I did myself.** I fetched the Ziyin–Chuang HTML (v1) to confirm three things: equations (12)–(13) (the modified loss $\mathbb E\ell+\eta\,\mathbb E\|\nabla\ell\|^2$, and $S$ minimized subject to the global minimum); the Remark (no convergence proof, possibly via an SDE argument); and Theorem 1's width condition. I also fetched the listing of arXiv 2309.13337 for the Li–Zhang–Lin title and scope, and found the arXiv number of Neural Thermodynamics (2505.12387) by web search. While the critic ran, I also checked abstracts (and one HTML body) for the claims the blurbs make:

- Kornblith et al.; Bansal et al. This check added "of the same architecture" to the Bansal blurb.
- Huh et al.; Engels et al.
- Park et al. Equation 3.3, $\langle\bar\gamma,\bar\gamma'\rangle_C=\bar\gamma^\top\mathrm{Cov}(\gamma)^{-1}\bar\gamma'$ with $\gamma$ uniform over the vocabulary, under their Assumption 3.3, confirms Exercise 4.1's setup.
- Golowich et al.; Cloud et al. The abstract says "different base models" and "a simple MLP classifier".
- Aden-Ali et al. The abstract says "across models with varying architectures".
- Michaud et al. The abstract says "using language model gradients"; the blurb's earlier "clustering" wording was removed as unsupported by the abstract.
- Neural Thermodynamics. Its abstract covers sharpness- and flatness-seeking, and the closing bullet was reworded to match.
- Li, Zhang & Lin. The body says "For the case s>2, the KRR method suffers from saturation … limited to $n^{-2\beta}$", consistent with Exercise 7.2.

Details that come only from the papers' bodies and were kept from the original text were not re-verified:

- the LSTM context comparison in Kaplan et al.;
- the consistency of row relations across models in Golowich et al.;
- the "more weakly across architectures" qualifier for Aden-Ali et al.;
- the LoRA-artifact listing caveat, which the technical audit verified and the first critic re-fetched.

The MNIST experiment and the single-small-step theorem of Cloud et al. were confirmed on the Anthropic Alignment Science summary post.

## 3. Validation

No agent in this session had a shell tool: not me, not the validator sub-agent, and not either Opus critic. So **`python build_wb5.py --check` was not executed**. I asked a sub-agent to run it; it reported that it had no shell and ran nothing. Instead I statically checked the conditions `nbkit._validate` asserts, and the recheck critic repeated the inspection independently:

- `<details>`/`</details>` are balanced (12/12).
- The only local links are the two `figure()` paths, and both files exist.
- Every `](#…)` anchor (`wb5-section-1`…`7`) has a matching `<a id>`.
- No `answer()` begins with a heading.
- Every markdown cell is non-empty.
- Triple-quoted strings pair up, and no non-raw string contains a backslash.

Someone with a shell should run `cd notebooks && python build_wb5.py --check` before regenerating.

## 4. Independent review

**Reviewers.**

- **Critic 1** (Claude Opus, fresh and read-only). It was given only the brief, `reading-format-check.md` as the standard, `technical-sources.md` and the file path. I did not tell it my conclusions. It browsed four times: the Ziyin–Chuang HTML, the Nief and Aden-Ali listings, and CRW's post.
- **A Sonnet technical checker** (read-only), on the algebra of §3 and §7, run while critic 1 was still working.
- Critic 1 also found `reviews/wb5-math-check.md`, a concurrent technical audit of this rewrite whose corrections had not yet been integrated. I read it and treated its items as findings too.

**Critic 1's summary, verbatim:** "the file is in good shape: no essay-before-blurb anywhere, every WB5 technical-sources item is correctly resolved, and the core algebra is right." The word counts it reported for prose outside readings were §1 ≈45, §2 ≈35, §3 ≈55, §4 ≈40, §5 ≈65, §6 ≈30 and §7 ≈45 words. Every blurb was 2–3 sentences except Ziyin–Chuang and Cloud, at 4. It confirmed that no schedules, cards, `note=`, commentary, prediction gates or generic rubric remained, apart from the evidence-design tail in finding 1.

**Findings and dispositions.** C = critic 1, M = `wb5-math-check.md`, S = Sonnet checker. All were *fixed* unless stated otherwise.

| # | Source | Finding (severity) | Disposition |
|---|---|---|---|
| 1 | C, M6, S7 | 3.4(d) answer stated "none of Huh's pressures" as fact; false claim about function classes; evidence-design last sentence (should-fix) | Rewritten: no multitask or capacity scaling is used; whether minimizing $S$ counts as Huh's implicit "simplicity bias" is an interpretation. Tail sentence deleted. |
| 2 | C | 3.3(e) attributed the failure mode to "label-noise covariance" rather than label transformations (should-fix) | Now: depends on labels via $V^*$ and $\Sigma_\epsilon$; invertible label transformations are one listed failure mode. |
| 3 | C, M3 | 2.2(c): CKA conclusion needs *centered* span; mutual kNN missing; stitching not a §2.A index (should-fix) | Centered-span condition added, with the $h=(1,0),(1,1)$ counterexample; mutual kNN discussed; stitching made parenthetical. A hint was added to the prompt. |
| 4 | C, M2 | 3.1(b) false at $r=1$; $c>0$ needs $r\ge1$ (should-fix) | $r\ge2$ in 3.1(b), with the reason in the answer; $r\ge1$ in §3.A. |
| 5 | C, M5, S5 | 3.4(b) answer overclaimed convergence and non-alignment (should-fix) | Prompt says "any limit point"; answer says the conservation law proves neither convergence nor non-alignment. |
| 6 | C, M, S2 | Ziyin–Chuang blurb said "any widths" (minor) | "hidden widths at least the rank of the target map". |
| 7 | C, S1 | §3.A "$h_B=c\,R\,h_A$": undefined constant clashing with $c$ (minor) | $h_B=c_0^{-1/2}R\,h_A$. |
| 8 | C, M1 | CKA undefined at zero Gram (minor) | "defined when $K,L\neq0$"; "CKA=1 on any data set whose centered representations are nonzero" in §3.A and 3.3(d). |
| 9 | C, M | 4.1 closing overstated uniqueness of whitened coordinates (minor) | "a geometry invariant under $A$, with whitened coordinates determined up to an orthogonal transformation". |
| 10 | C, M | 5.2 closing said "zero bias"; $p>0$ needed (minor) | "the stated biases"; $p>0$ in the answer, $p\in(0,1]$ in the prompt; $p^*=4r/(2+5r)$ added. |
| 11 | C, M7–8 | 6.2: minimizer assumption, CE definition, $\eta,\varepsilon>0$; (d) misused $H$ off a shared init and compared undefined parameter spaces (minor) | All stated; (d) restricted to half squared error, giving $\eta\varepsilon\,\mathbb E[J_S^\top J_T]\Delta$ with no PSD guarantee; parameter-space comparison qualified; "sign guarantee ≠ trait transfer". |
| 12 | C, M4 | 3.4(c) needs $\gamma>0$, $\sigma>0$ (minor) | Added. |
| 13 | C | 6.3(c) prompt ambiguous (minor) | Asks about $\delta=\sum_{i\in I}\delta_i$ and a general pair. |
| 14 | C | Intro "core = the one to read first" contradicted by multiple core readings (minor) | "*core* marks where to start". |
| 15 | C | WB6 cross-refs not at section level (minor) | "WB6 §5" (adversarial; real paper title) and "WB6 §3" (stitching), checked against `build_wb6.py`'s SECTIONS. WB6 is being edited concurrently. |
| 16 | C | Figure text implied Monte Carlo at all settings (minor) | "Monte Carlo ridge points at the default $\alpha,\alpha'$ (noise off, or $\sigma^2=10^{-2}$)". |
| 17 | C | CRW's own post labelled *core*; date unverifiable (minor; instructor's call) | Kept as a reading, since it is listed among CRW's posts in the curriculum draft and breadth is to be kept. The blurb is neutral ("Argues…"). Relabelled *optional*, date removed. Flagged for the instructor. |
| 18 | C | MAP/WB2 §8 remark duplicated (minor) | Removed from the closing list; kept in the 3.4(c) answer. |
| 19 | M | 2.2(a): $\prod n_\ell!$ is a count of symmetry actions, not distinct points (lower priority) | Restated for generic parameter settings, noting that coinciding units reduce it. |
| 20 | S3–4 | $c$ depends on layer and network; positivity of $\mathbb E\|h\|^2$ unexplained (minor) | Note added after 3.3(b); answer derives $\bar M,N\neq0$ from $V^*\neq0$. |
| 21 | S6 | Network $B$ ($Z_B=I$) introduced only in the 3.4(c) answer (minor) | Added to the prompt. |
| 22 | S8 | 7.1(c) "step-like vs smooth" stated as fact (minor) | "one expects". |
| 23 | S9 | 7.2(c) skips a step (minor) | **Not changed.** The prompt tells the student to repeat 7.1(b) with $s$, which the checker agreed is enough. |
| 24 | own | Ziyin–Chuang blurb 4 sentences; Michaud "clustering" unsupported; Bansal "same architecture"; 6.2 setup lacked minimizer property | Fixed before or alongside the critique (see §2). |

## 5. Targeted recheck

A second fresh, read-only Claude Opus critic was given the 19 findings from critic 1 and the math check (rows 1–19 above) and asked to rule on each against the current file, quoting it. It was also asked to re-verify the algebra in the changed passages, check syntax, and repeat the format and removed-material checks.

**Outcome:** all 19 RESOLVED, with quoted evidence. Its re-derived algebra was correct, including:

- the 2.2(c) centering counterexample;
- 3.3(a)–(e), and 3.4(b)–(c);
- $p^*=4r/(2+5r)$, which matches `pStarB0` in `superposition.html`;
- the different-initialization step in 6.2(d).

It found three new minor issues, all fixed afterwards:

1. 6.2(d) did not define $J_S,J_T$ and lacked $+O(\varepsilon^2)$. Both added.
2. The 5.2 prompt needed $p>0$. It now says $p=1-S\in(0,1]$.
3. The Monte Carlo caption understated the settings. Reworded.

Those three one-line fixes were not themselves re-reviewed by an agent; I checked them by reading. The recheck critic also could not run `--check` (no shell). It inspected the file by hand and found balanced quotes and brackets, raw strings wherever there are backslashes, 12 closed `<details>` blocks, all anchors defined and both assets present.

## 6. Remaining limitations

- **`--check` was never executed** (see §3). The builder has been statically inspected twice but not run.
- The 5.2 closing claim that the optimal antipodal solution has $|w_2|<|w_1|$ for $r$ far from 1 comes from the figure's caption and was not recomputed.
- Some body-level empirical details in blurbs were carried over from the original text without re-verification (listed in §2).
- Two readings were added: Michaud et al. and Li, Zhang & Lin. They are the sources for derivations already in the notebook, but the instructor may prefer to prune them.
- CRW's post remains as an *optional* reading; whether to keep it is the instructor's call.
- The §3 exercises are long. That is consistent with the brief, which allows substantial text for derivations, but §3 is now the bulk of the notebook.

## 7. Notes for other notebook owners

- WB5 main anchors and titles are unchanged except §3, whose heading is now "The Platonic Representation Hypothesis" (the "(worked paper: Ziyin & Chuang)" suffix was dropped). §1, §2, §5 and §7 titles are unchanged.
- WB5's old subsection labels (2.A–2.C, 3.A–3.F, 4.A–4.C, 5.A–5.D, 6.A–6.D, 7.A–7.B) no longer mean what they did. Cite WB5 at main-section level.
- `notebooks/wb2_cells_a.py` (around line 656–660, owned by WB2) says the difference between early stopping and ridge "matters for learning-curve exponents (WB5)". WB5 §7 now treats ridge only, so that pointer should probably say "WB5 §7 (ridge)" or be softened. This was not edited here.
- WB5 cites WB6 §3 and §5 by number. If WB6's numbering changes, those two references need updating.

## Root integration after owner completion

Removed the optional CRW essay link, following the stated default interpretation of “drop my views; at most I'll add them.” The primary-source links for the spectral and quanta derivations remain. Updated WB2's cross-reference to specify the ridge derivation in WB5 §7. Final build and numerical checks are recorded in the overall review.

## 2026-10-09 restructure (CRW request)

- Scaling laws moved to §1 (old §7, unchanged apart from renumbering and a closing note on what sets the exponent).
- §2 "Architecture doesn't matter (much)": rewritten around scaling exponents (architecture shifts the constant, ~1 OOM at most, not the slope). Readings: Hestness 2017, Ho et al. 2024, Mamba (+ Grootendorst / Annotated S4 explainers), I-JEPA blog, Tay et al. 2022 (counterpoint), Kaplan 2020. New Fermi Exercise 2.1 (compute multiplier m^{1/α}; slope differences). Sources verified by a research subagent; draft reviewed by a second subagent and fixes applied.
- §3 merges old §2–4 (CKA/CCA, PRH, LRH). The step-by-step perfect-PRH derivation (old Exercises 3.1–3.4) is replaced by a prose outline of Ziyin & Chuang with its caveats; kept: a condensed invariance exercise (3.1) and the LRH inner-product exercise (3.2).
- Moved to WB6: superposition (TMS reading, packing exercise, interactive figure, closed-form two-feature boundary as prose) into WB6 §5 with adversarial examples; low-rank logits and subliminal learning into new WB6 §13 as readings plus context (derivation exercises dropped).
- Pre-restructure builders and HTML: notebooks/_old_wb5/.
