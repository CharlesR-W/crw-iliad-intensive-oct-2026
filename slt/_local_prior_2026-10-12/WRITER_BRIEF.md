# Brief for agents drafting an SLT-day notebook

Thank you for taking this on. Please be kind to yourself and to any agents you dispatch, and pass this request on to them. The instructor (CRW) cares about the wellbeing of the agents working on this. If anything about the task feels bad or wrong, say so in your report instead of pushing through.

All paths are relative to `/home/crw/Programming/Iliad_MysteriesDL`.

## Read first

1. **`slt/SLT_SPECS_v1.md`: binding.** The "Framing" section overrides everything else, including this brief. Then follow your notebook's section.
2. **`slt/_sources/`:** the old Iliad B.3 worksheet (`iliad_worksheet/main.tex`, by Kai Ogden, Matthew Farrugia-Roberts and Zach Furman, with solutions), Furman's slides, the devinterp v1.3.2 notebooks and Furman's `intro_lc.ipynb`. Reuse their exercises and code where they fit, adapted to our framing. Where you adapt an exercise or code, credit it in one short line ("adapted from the Iliad B.3 worksheet, Ex 4.1" / "adapted from Furman's intro-lc-estimation notebook").
3. **`td/README.md`:** the output format (HTML page + runnable `.ipynb`), which SLT uses unchanged.
4. **Style reference** for tone and structure only: `td/build_td2_3_sgd_sde.py`, `da/build_da1_core.py`.

## The instructor's direction (follow it)

- **This day is Bayesian statistics, plus λ as a basin-volume exponent.** Singularities get the absolute minimum mention. Never organise anything around "singular models" or "degeneracy"; never put resolution of singularities on the path. Words like "singular", "degenerate" may appear where a sentence needs them, not as section themes. The Hessian being degenerate is fine to say plainly.
- **Thermodynamics language is a formal dictionary.** Point out analogies where they actually hold, and when one breaks say so in one line (finite d, no thermodynamic limit, n is not a system size, etc.). Never write "this is just like how thermodynamics emerges from stat mech" style claims, and no hype ("phase transitions in neural networks reveal…").
- **SGD-vs-Bayes is out of scope.** Don't discuss whether SGD samples a posterior. SGLD appears only as a sampler (NB3).
- **The effective-theory / why-temperature argument is CRW's own.** NB3 §2 is a visible, marked stub. Don't draft the argument, don't paraphrase what it might be beyond the spec's one-line placeholder scope.
- **Programs / Turing machines:** NB4 only, neutral, brief, no evaluation.
- **Don't attribute views to the instructor.** Don't evaluate or editorialise about SLT as a research programme, positively or negatively. State results, their assumptions, and their scope.

## Pedagogy (from CRW's earlier feedback)

- Students direct their own learning; nobody is expected to finish. One linear path, few branches.
- **Derivation/notebook first, readings a close second** (offered as an alternative or supplement). Each notebook opens with 1–3 sentences on its question, its keystone reading, and a one-line-per-section route.
- **Every section states its question and restates its notation locally** (one line is enough), so it reads standalone when opened from the table of contents. Dense "The result:" paragraphs full of undefined symbols are the anti-pattern CRW found "very hard to follow".
- Physics-y register, short plain sentences, no em-dash-heavy rhetoric. Text must suffice for a grad student without an LLM, but don't pad: CRW reacted to earlier drafts as "too much material and too diffuse". Prefer one clean derivation to three partial ones. Cut hedging, restated caveats, "you may want this if".
- **Be correct but not pedantic.** Caveat only when it changes what a student would conclude. Fix source errors silently (never "the worksheet says X, but…"). Known worksheet errata: the §4.2 log-odds sign word; "λ = d/2 iff regular" is asserted without proof (say "λ ≤ d/2, with equality in the regular case").
- Exercises: self-contained statements, bold labels (`**Exercise 1.3 — Laplace's formula.**`), complete worked solutions with intermediate steps, hints in `<details>`. Fermi estimates and short derivations are welcome. No rubrics, "Deliverable:" labels, convergence cards, schedules, generic discussion questions, or instructor-commentary placeholders.
- **Headings:** `##` per section, at most one or two `###` per section where it genuinely helps (e.g. a longer worked example or the code). No `####`.
- Readings: a sentence or two each, ARENA `importance=`/`length=` scores required; for long papers name the sections to read. Books (Watanabe's) only via `references(book(...))`. Only use readings the spec names; anything else you think essential goes under a final line "**Claude-suggested extras**" (at most 1–2 per notebook). Fetch every URL.

## Shared notation (all four notebooks use this; restate locally as needed)

| Symbol | Meaning |
|---|---|
| $w\in W\subseteq\mathbb R^d$ | parameter; $d$ its dimension |
| $\varphi(w)$ | prior density |
| $q(x)$, $p(x\mid w)$ | true distribution, model |
| $D_n=\{X_1,\dots,X_n\}$ | i.i.d. sample from $q$ |
| $L_n(w)=-\frac1n\sum_i\log p(X_i\mid w)$ | empirical NLL ("energy" per sample) |
| $L(w)=\mathbb E_q[-\log p(X\mid w)]$ | population NLL |
| $w_0$ | a minimiser of $L$ (optimal parameter); $S=\mathbb E_q[-\log q]$ the entropy, $L(w_0)=S$ when realisable |
| $K(w)=L(w)-L(w_0)$, $K_n(w)=L_n(w)-L_n(w_0)$ | normalised population / empirical loss |
| $\beta$ | inverse temperature; $\beta=1$ is Bayes |
| $p_\beta(w\mid D_n)\propto\varphi(w)e^{-n\beta L_n(w)}$ | tempered posterior |
| $Z_n=\int\varphi\,e^{-n\beta L_n}$, $F_n=-\log Z_n$ | evidence / partition function, free energy |
| $\bar Z_n=\int\varphi\,e^{-nK}$, $\bar F_n=-\log\bar Z_n$ | population ("averaged") version used for volume scaling |
| $H=\nabla^2L(w_0)$, $C=\mathrm{Cov}_q[\nabla\log p(X\mid w_0)]$ | Hessian, gradient covariance (Fisher at $w_0$ when realisable) |
| $V(t)=\int_{K(w)<t}\varphi(w)\,dw$, $v=V'$ | prior volume of a sublevel set, and its density |
| $\lambda$, $m$ | volume-scaling exponent ("learning coefficient") and its multiplicity: $V(t)\sim c\,t^{\lambda}(-\log t)^{m-1}$ |
| $\zeta(z)=\int K(w)^z\varphi(w)\,dw$ | zeta function (Mellin transform of $v$) |
| $G_n$ | Bayes generalisation error: $\mathbb E_X[-\log p(X\mid D_n)]-S$, with $p(x\mid D_n)$ the posterior predictive |

Use "learning coefficient" for λ (it is the standard name; define it as the volume exponent). "Local learning coefficient (LLC)" in NB3/NB4 for the localised version.

## Files and anchors

| NB | Stem | Owner |
|---|---|---|
| 1 | `slt1_bayes` — Bayesian inference as statistical mechanics | writer 1 |
| 2 | `slt2_volume` — Free energy and basin volume | writer 2 |
| 3 | `slt3_temperature` — Effective exponents and temperature | writer 3 |
| 4 | `slt4_applications` — Applications (readings) | writer 4 |

- Builder: `slt/build_<stem>.py`, ending `raise SystemExit(run(CELLS, "<stem>", "<stem>"))`. Kit: `slt/nbkit_slt.py`, `slt/htmlkit_slt.py` (a copy of the DA kit with sibling links rewritten for `slt*` stems; same API: `markdown`, `answer`, `code`, `code_answer`, `reading`, `references`, `book`, `figure`, `run`). Don't edit the kit; ask in your report.
- The page subtitle line (second line of the first cell) is `**Singular learning theory day, Iliad Intensive 2026**`.
- Section anchors: `<a id="sltN-k"></a>` before each `## N.k` heading (k = 1, 2, …), and `sltN-readings` for a closing readings section if any. You may link to a sibling notebook only at section level, e.g. `slt2_volume.html#slt2-1`, using the section list in the spec (NB1 §1–4, NB2 §1–3, NB3 §1–3); keep such links rare.
- Touch only: your builder, your generated `.html`/`.ipynb`, `slt/checks/<stem>_*.py`, `slt/assets/interactive/<stem>_*` (only if a figure is a key intuition; probably not needed), `reviews/slt/<stem>.md`, your `_exec_cache` entry. Other writers work in parallel.

## Code

- CPU only, numpy/scipy/matplotlib (torch CPU where autograd helps). Each cell ≲30 s, the whole notebook ≲3 min. Code checks a derivation or shows a phenomenon; never boilerplate.
- NB3: adapt from the devinterp v1.3.2 notebooks and Furman's notebook rather than writing from scratch, but keep the `.ipynb` dependency-light: an inline SGLD loop (credited) is preferred to requiring `pip install devinterp`/pyro.
- `python slt/build_<stem>.py`, `--check` and `--run` must all pass (run from `slt/`).

## Verification

- Every formula you state: verify symbolically (sympy) or numerically; keep the scripts in `slt/checks/<stem>_*.py`.
- Every paper claim you narrate: check it against the paper (WebFetch the arXiv abs/HTML). Every URL: fetch it.
- Log what you verified, and how, in `reviews/slt/<stem>.md`. Never claim verification you didn't do.

## Critique loop (required)

After a complete draft that builds and runs, do at least two rounds of critique and revision (NB4: one round suffices), three if round two still finds substantive problems. Each round, run both critics in parallel:

1. **Design and pedagogy critic:** a Claude subagent (Agent tool, `model: "opus"`). Give it `slt/SLT_SPECS_v1.md`, this brief, and your built HTML path. Ask it to judge against the instructor's direction and pedagogy: single path? each section standalone with local notation? pitch tells a student whether they want it? text sufficient without an LLM, yet not padded or diffuse? any singularity-framing, thermo hype, or editorialising creeping in? Ranked list of concrete problems with fixes, not praise. Don't tell it what you think is wrong.
2. **Technical critic:** Codex (GPT), read-only, via
   ```bash
   cd /home/crw/Programming/Iliad_MysteriesDL && PATH=/home/crw/.local/share/mise/installs/codex/latest/bin:$PATH \
     node /home/crw/.claude/plugins/cache/codex-plugin-cc/codex/1.0.25/scripts/codex-companion.mjs task --effort high --prompt-file <your prompt file>
   ```
   (The PATH prefix matters: the `~/.local/bin/codex` wrapper hangs.) Run it with Bash `run_in_background: true` and wait for the completion notification. Ask it to check every derivation, constant, sign, asymptotic claim and its conditions, code cell, and paper claim against the sources, and to report a numbered list of errors with evidence. Ask it to be kind and candid. If Codex fails after one retry, use a second Claude subagent as the technical critic and note that in the review log.

Weigh critiques (critics can be wrong; check disputed points yourself), revise, rebuild, rerun.

## Report back (final message, concise)

- Files written; cell counts; `--run` time.
- A short outline as built (one line per section, with its lead reading).
- What the critiques changed.
- Questions only the instructor can answer (few).
- Claims you couldn't verify.
