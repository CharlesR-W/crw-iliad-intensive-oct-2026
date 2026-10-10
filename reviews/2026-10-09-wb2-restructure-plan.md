# WB2 restructure plan — 2026-10-09

## CRW's feedback (paraphrased from the session, authoritative)

- Put the "old-fashioned statistical learning" together as section 1: risks, bias–variance, regularization.
- The regularization part is too short to explain anything; at minimum link the Goodfellow–Bengio–Courville *Deep Learning* book (ch. 7).
- Information bottleneck is too small to stand alone; make it a (long) subsection of section 1. Rate–distortion is good. Don't open by complaining about mutual information (but keep that material).
- Theorems: start with NFL, then UAT; a short PAC-Bayes intro would be good; VC is already there; the Zhang et al. reading is good.
- Overfitting and double descent is fine as is; may be enlarged or stay small-ish.
- Bayes vs MLE, C vs H, and Fantastic Generalization Measures are one sub-story and go together.
- Mention kernels and the kernel trick somewhere.
- Mention the traditional supervised / unsupervised / etc. breakdown of learning problems.
- **Readability:** "I'm having a very hard time following — it doesn't do enough to introduce notation or context — make things relatively standalone." Every section and subsection must say what question it addresses, why it matters, and define its symbols before use (restate, don't just point to "§3.B").

## New structure

Exercise groups are h4 headings numbered `X.Y.k` (e.g. `#### 1.D.3. L1 and sparsity`), with numbered parts; cross-references look like `1.D.3.2` (group 1.D.3, part 2).

### Opening
Title, how to use, route table, short notation primer.

### 1. Classical statistical learning (anchor `wb2-section-1`)
- **1.A Learning problems, risks, and two kinds of generalization.** Kinds of learning (supervised, unsupervised, self-supervised, semi-supervised, reinforcement; generative vs discriminative; batch vs online). The supervised i.i.d. setting; loss, risk, ERM, θ*. Exercise 1.A.1 (old 1.A.i).
- **1.B Ordinary least squares, exactly.** Linear model + SVD primer. 1.B.1, 1.B.2 (old 1.B.i–ii).
- **1.C Bias–variance.** 1.C.1 (old 3.A.i), 1.C.2 (old 3.B.i).
- **1.D Regularization.** NEW taxonomy intro linking GBC ch. 7 sections. 1.D.1 prior (old 4.A.i); 1.D.2 constraint (old 4.B.i); 1.D.3 NEW L1/sparsity; 1.D.4 spectral filter & early stopping (old 4.C.i + figure); 1.D.5 NEW input noise and augmentation; 1.D.6 dropout (old 4.D.i); 1.D.7 flatness (old 4.E.i); 1.D.8 Dinh reading + question (old 4.F).
- **1.E Kernels and the kernel trick.** NEW.
- **1.F Information bottleneck and rate–distortion.** NEW opener: information-theory primer, rate–distortion theory (1.F.1 NEW), IB as rate–distortion (1.F.2, old 6.E promoted), then the information-plane debate (old 6.A–6.D as 1.F.3–1.F.7), readings (old 6.F) + optional VIB reading.

### 2. Doesn't some theorem make this impossible? (anchor `wb2-section-2`)
Pitch + what a PAC guarantee is. 2.A NFL (old 2.A); 2.B UAT (old 2.B); 2.C capacity: NEW short VC primer exercise + old 2.C.i; 2.D Zhang et al. reading + questions (old 2.D); 2.E NEW PAC-Bayes intro: bounds on the found (stochastic) predictor; Occam bound as special case; Gaussian posterior → weight-norm + flatness; readings.

### 3. Overfitting and double descent (anchor `wb2-section-3`)
3.A–3.C (old 5.A–5.C, figure); 3.D NEW benign overfitting: a high-dimensional junk tail makes the min-norm interpolant act like ridge (Bartlett et al. mechanism); 3.E what regime are we in (old 5.D); 3.F readings.

### 4. Bayes, MLE, and generalization measures (anchor `wb2-section-4`)
One story: local theory ½Tr(HΣ) → ERM sandwich and AIC/TIC → Bayes/Laplace/BIC → three estimands → does curvature cancel (C vs H) and SGD's stationary law → do any of these predict generalization in practice (Fantastic). 4.A (old 7), 4.B (old 8), 4.C (old 9). Rewrite openings for context; define H, C, "regular", log loss, TIC/AIC/BIC before use.

## Old → new section map
§1→1.A–1.B; §2→2 (+2.E); §3→1.C; §4→1.D; §5→3; §6→1.F; §7→4.A; §8→4.B; §9→4.C.
