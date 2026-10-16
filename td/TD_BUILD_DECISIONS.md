# Training Dynamics notebooks: build decisions (2026-10-08)

This file records CRW's (the instructor's) answers given on 2026-10-08 before drafting. Where it conflicts with `../TD_SPECS_v1.1.md`, this file wins.

## Authority, highest first
1. CRW's own topic note `../Training Dynamics` (a sketch of interests, not a rigid spec).
2. CRW's pedagogy (below) and the MDL editorial brief `../reviews/2026-10-08-editorial-brief.md`, which applies to TD too.
3. This file.
4. `../TD_SPECS_v1.1.md` (agent-written; CRW skimmed and roughly approved it). Use it as the content map, not as authority.
5. `../TD_LINKS_v1.md`: verified links and one-liners. Items 52–57 were written from memory and are unchecked. Item 5 (Kovachki–Stuart) has a suspected sign error in a printed constant. Check any claim before you narrate it.

## CRW's pedagogy (CRW's own words, lightly condensed)
> Students should be in charge of their own learning. I only want a small contextualizing lecture or two. The workbooks should be something like a set of exercises or readings I think would be very beneficial for students to do. For each section, present an idea of what the section is about, enough that students can tell whether they want to learn it. Then offer a discussion. The discussion is a default text for students, at a default physics-y level of rigor and focus, like what I prefer. I expect students to look at whatever they find interesting and discuss it with their own agents, so we do not need to cater to every taste. We just need to give enough context that students from different backgrounds know what there is to learn, and can decide what to learn and at what level of rigor. Not all students will have huge LLM subscriptions, so our texts should in principle be enough to teach each concept at the level I think is appropriate. We shouldn't totally skimp.

From the MDL brief, also binding here:
- Nobody is expected to finish. Breadth is fine if it is well organized.
- Derivations are mostly exercises with worked answers. Give enough setup, intermediate steps and hints (in `<details>`) to learn without an LLM, and link the original paper.
- Reading-focused parts give only a few sentences per reading (enough to decide whether to read it) plus the link. Don't replace papers with long exposition.
- No universal claim/evidence/falsification rubric. No "Deliverable:" boilerplate, no "commit to a one-sentence answer" prompts, no generic discussion padding. Keep a question only where it genuinely helps.
- Remove the convergence cards at 17:00, shared schedules, and "predict before you open the figure" instructions. A local "core" reading label is fine.
- Don't attribute views to the instructor. CRW adds their own positions.
- All solutions are available to all students.

## Section shape (replaces the spec's read / derive / go-further labels)
1. **Pitch:** what the section is about and why someone would want it, in a few sentences, plus "what you need" (prerequisites restated so the section stands alone).
2. **Default text:** physics-y exposition, sufficient on its own.
3. **Exercises:** markdown answer cells holding worked solutions, and code exercises where they help. Hints go in `<details>`.
4. **Readings:** short annotated list. Mark the core reading if there is one.

"(ask your agent)" is allowed only as a pointer for results beyond the section's scope. Never use it for material the section needs.

## Decisions
- **Code: yes.** Add small CPU code cells next to the key derivations so students can check them numerically and see the phenomenon. Use numpy and matplotlib; torch (CPU) is fine where autograd or Hessian-vector products are needed. Each cell should take under about 30 s and a whole notebook under about 3 min on a laptop. Use `code()` for demos and `code_answer(stub, solution)` for code exercises. Math derivations stay as markdown `answer()` cells. Every builder must pass `python build_X.py --run`.
- **Interactive figures:** at most three for the whole day, each standalone HTML in `assets/interactive/` (Distill-style, key intuitions only): the NB1 mode explorer, NB2 §3's self-stabilization toy plus catapult map, and NB2 §4's steepest-descent unit balls.
- **NTK:** keep the spec's coverage: the linear-net NTK, imbalance as the lazy/rich dial, and the general infinite-width NTK only as a pointer. The Physics of Deep Learning day (`/home/crw/Programming/IliadPhysicsLearning`) covers NTK and infinite width, so link to it as the escape hatch.
- **Dropout (geometric-mean ensemble), layer cushion / noise robustness / basin size, and Kadkhodaie's geometry-adaptive bases** stay as readings in the further-readings notebook.
- **Keep all agent-added topics:** normalization, weight decay, spikes and clipping (NB2 §5); catapult and warmup/WSD; critical batch size and Adam's square-root scaling; fine-tuning and RL update dynamics (as readings).
- **NB1 "Saxe-y" heuristics.** CRW suspects some of them amount to "propose a model with planted structure, then recover that structure". Err on the side of including what the previous iteration of this course covered (below). For each heuristic, say plainly which part is a general consequence of the derivations and which part follows from structure planted in the model or data. Keep the agent-added extras (semantic development, dynamical isometry, Advani–Saxe generalization dynamics) short, with the same honesty.
- **Moment bias:** drop the "why the moment story doesn't get you far" question entirely. CRW doesn't want to argue it. §7 ("what is learned first") stays as neutral exposition.
- **SGD as sampling a posterior:** out of scope (MDL WB2 §4.B ("Does curvature cancel?") and its `sgd_stationary_vs_posterior.html` figure cover the stationary-vs-posterior comparison; cross-link it). Morse–Bott and Witten are out too.
- **Guest lecture (Daniel Kunin):** don't feature or evaluate his work specially. Nothing depends on the lecture.
- **Lectures:** produce outlines and speaker notes for L1 and L2 after the notebooks exist.

## What the previous course iteration covered (Corlouer, B.4; see `../Iliad B-track - current materials and prereqs.md`)
Deep linear nets, squared loss, teacher SVD. (1) The landscape (Achour et al.): mode decoupling, critical points, no spurious local minima, a gauge-symmetric minimum manifold. (2) Gradient flow: conserved balancedness, Ẇ = K[M−W] (the NTK operator). (3) The rich regime (Saxe 2014): logistic mode ODEs, saddle-to-saddle dynamics, low-rank bias. (4) The lazy regime: frozen NTK, exponential convergence. (5) The mixed regime (Tu–Aranguri–Jacot 2024): lazy-to-rich transition, grokking link. (6) Bonus: SGD noise as a Langevin SDE, Fokker–Planck, a Boltzmann law, flat minima. Covered on slides only: general NTK, edge of stability, SLT link, neural race reduction, Alternating Gradient Flows.

Known problems in the old version, to avoid: the lazy section contradicted itself (the NTK changed by about 100% in its own model); "balanced init" was conflated with an imbalance parameter; alignment was assumed and never derived; depth > 2 and EoS were promised but missing.

## Files (all under `td/`)
| Stem | Content | Spec section |
|---|---|---|
| `td1_linear_networks` | NB1: Saxe dynamics, landscape, NTK of linear nets, lazy/rich/mixed, heuristics | NB1 |
| `td2_1_flows` | GD as flow + corrections, momentum | NB2 §1 |
| `td2_3_edge_of_stability` | EoS, self-stabilization, central flows, adaptive, EoSS, catapult, warmup | NB2 §3 |
| `td2_2_sgd_sde` | SMEs, linear scaling, critical batch, Adam scaling, OU in a basin | NB2 §2 |
| `td2_4_optimizer_geometry` | steepest descent under norms, natural gradient, Muon, muP | NB2 §4 |
| `td2_5_normalization_weight_decay` | scale invariance, WD equilibrium, spikes, clipping | NB2 §5 |
| `td2_6_curvature` | GGN, bulk + outliers, tiny subspace, basin commitment | NB2 §6 |
| `td2_7_learned_first` | linear-statistics-first, spectral bias, higher cumulants | NB2 §7 |
| `td2_8_further_readings` | NB2 §8 readings + day index | NB2 §8 |

Builders: `build_<stem>.py`, which import `nbkit_td`. Never hand-edit the generated `.html` or `.ipynb` files. Review logs go in `../reviews/td/<stem>.md`.

## Output format (2026-10-09, CRW's choice, matching the MDL day)
Each builder writes two files. The exercises/solutions `.ipynb` pairs are no longer generated. The last versions are kept, unmaintained, in `_old_ipynb/`.
- **`<stem>.html`**: one reading page in the MDL pages' style (template, CSS and KaTeX loader copied from `../notebooks/nbkit.py` into `htmlkit_td.py`). Every `answer()` is a folded "Solution" after the cell it answers. `code()` cells show their code and captured output: stdout, and matplotlib figures as PNGs. `code_answer()` cells show the stub, then a folded "Solution (code)" with the solution code and its output. Interactive figures are embedded in an iframe below their `figure()` link (once per page).
- **`<stem>.ipynb`**: the runnable notebook. It has the stubs and blank answer cells, and points to the page for solutions.
- Outputs come from executing the solution code at build time. They are cached in `_exec_cache/`, keyed on a hash of the code, so `--check` never executes. `--run` still times every solution cell against the 60 s budget.
- Links between notebooks may be written in the old `_solutions.ipynb#anchor` form. The kit rewrites them to `<stem>.html#anchor` on the pages and to `<stem>.ipynb#anchor` in the notebooks. New text should link `<stem>.html#anchor`, and should not refer to "the solutions version" of a notebook.
- See `README.md` for students and maintainers.

Conventions shared across NB2 (from the spec): loss $L$, Hessian $H$, step $\eta$, batch $B$, per-example gradient covariance $\Sigma$, $G=\nabla L$, sharpness $S=\lambda_{\max}(H)$, continuous time $t=k\eta$ (so gradient flow is $\dot\theta=-\nabla L$). NB1 uses $\tau=1/\eta$. Each NB2 file must stand alone: restate anything borrowed from another section in its "what you need" box.


## 2026-10-14: notebooks 2.2 and 2.3 swapped

CRW: the edge of stability (and the catapult, warmup and WSD that live with it) comes *after* the small-step expansions. NB2 is now §1 flows, §2 SGD as an SDE, §3 edge of stability. Files, anchors (`td22-*`/`td23-*`) and section numbers were swapped accordingly; the lecture L2 follows the same order.

## 2026-10-14: readings first, derivations following

CRW, 2026-10-14: readings first, derivations following. Each notebook now opens (after its short intro and route) with a single `## Readings` section that collects every paper of the notebook, grouped by section, with its ARENA scores and a one-sentence blurb; the `##` sections after it are the derivations and walkthrough. This replaces "paper first per section" from the 2026-10-09 streamline (`../reviews/2026-10-09-td-streamline-brief.md`); the rest of that brief (short pitches, no new papers, flat headings) still holds. Mechanics: the per-section `reading()` cards moved into the Readings block (none added or dropped; trailing reading lists merged into it), body sections point back to it in one line, and the `### Optional: annotated derivations and walkthrough` sub-headings were flattened into a bold "**Derivations and walkthrough.**" lead-in, keeping their `-walkthrough` anchors. Every existing anchor survives. The kit renders a `##` titled "Readings" open by default (`OPEN_SECTIONS` in `htmlkit_td.py`), so the papers are the first thing a reader sees.
