# MDL restructure brief: keystone + annotated trail (2026-10-09, afternoon)

This supersedes `2026-10-08-editorial-brief.md` wherever they conflict. CRW's request, in their words (dictated; lightly cleaned):

> Re-organize MDL notebooks to each have 1 key section with extra stuff. The 'key' one should be as directly related as possible to 'mysteries of deep learning', with the rest as further readings which are a little less apposite. Roughly 1–2 core ideas per notebook, with a small number of readings. The rest can be organized similar-ish to how it is now (but with much shorter explanations). Each section should have a concise summary for why it's there. Streamline the notebooks, especially the text descriptions.
>
> This is too much material and too diffuse. I don't want to toss it all, but I'll live if stuff gets cut — it's fine to make some things just references or things to learn about. New preferred format: the keystone reading for each WB, then turn the current subsections into something much more like **lightly annotated reading lists**. Each section gets a ~1-sentence description of its content/argument/purpose, and likewise each paper. We're sketching out a trail. It's fine to keep the exercises, but put them in subsections named for this.

## New workbook titles and keystones

| WB | File (keep stem) | New title | Keystone |
|---|---|---|---|
| 1 | `01_intelligence_then_and_now` | How Deep Learning Changed How We Think about Intelligence | Sutton, *The Bitter Lesson*; Ngo, Chan & Mindermann, *The Alignment Problem from a Deep Learning Perspective* |
| 2 | `02_statistical_learning` | How Deep Learning Changed Statistical Learning Theory | Double descent and the vacuity of bounds. Link Lawrence Chan's LessWrong posts: [The paper that killed deep learning theory](https://www.lesswrong.com/posts/ZvQfcLbcNHYqmvWyo/the-paper-that-killed-deep-learning-theory) (Zhang et al. 2016; 1.8k words), [The other paper that killed deep learning theory](https://www.lesswrong.com/posts/zcGmdQHX66NhC69v6/the-other-paper-that-killed-deep-learning-theory) (Nagarajan & Kolter 2019, uniform convergence; 2.3k words), and the follow-up [Maybe I was too harsh on deep learning theory](https://www.lesswrong.com/posts/6SRq7mZ97Dwuavwb6/maybe-i-was-too-harsh-on-deep-learning-theory-three-days-ago) (infinite-width/NTK/NNGP line turned out more interesting; 1.2k words). Plus one double-descent paper. |
| 3 | `03_why_llms` | Why Did LLMs Become the Dominant Paradigm? | Gwern, *The Scaling Hypothesis* (my pick; CRW named none) |
| 4 | `04_inductive_bias` | Deep Learning as Program Synthesis | Zach Furman, *Deep learning as program synthesis* (LessWrong, Jan 2026) |
| 5 | `05_universality` | Deep Learning and Universalities: Scaling Laws and Architectures | Kaplan et al. 2020, *Scaling Laws for Neural Language Models* |
| 6 | `06_mysteries_myriad` | Mysteries of Deep Learning | none: the whole WB is the annotated catalogue |

Keep file stems (TD notebooks and other pages link to them).

## Page shape

```
# Workbook N — <new title>
<1–2 sentences: why this WB is in a "Mysteries of Deep Learning" day.>
<route: one line per section, linking anchors>

## 1. Keystone: <short name>          (anchor wbN-keystone)
<Why it's here: 1–2 sentences.>
<1–2 keystone readings via reading(), each blurb ~1 sentence.>
<The 1–2 core ideas: one short paragraph each (≲120 words), enough that a student who reads nothing else gets the point. No derivation chains here.>
### Exercises: <topic>                (optional; only if an existing exercise directly serves the keystone)

## 2. <Trail section title>          (anchor wbN-section-2, etc.)
<~1 sentence: what this section covers / argues / why it's on the trail.>
<lightly annotated reading list: reading() blurbs of ~1 sentence (≲35 words) each>
### Exercises: <topic>                (existing exercises kept, with their folded solutions)
...
```

- Sections are `##`, exercise subsections `###` (nbkit folds both). Avoid `####` unless an exercise set truly needs it.
- Every section and subsection opens with its one-sentence purpose. No other prose at section level beyond that sentence and, if unavoidable, one short paragraph of context.
- Readings: ~3–6 per trail section, each ~1 sentence. Prune to the ones that directly bear on the section's point; demote others to a single trailing line "Also: [title](url); [title](url)." or drop them. Keep ARENA ratings (`importance=`, `length=` required by `reading()`); books only via `references(book(...))`.
- Exercises: keep the good ones (especially those CRW asked for: Bellman/Q-learning in WB1; derivations in WB2; Fermi estimates in WB3/WB5). Put them in `### Exercises: <topic>` subsections. Each exercise must be self-contained: move whatever notation/setup it needs into the exercise statement itself (one or two lines), since the surrounding narrative prose is going away. Trim solutions where they are padded, but keep them correct and complete. Exercises that only made sense inside a long derivation chain you are cutting may be dropped or turned into a "things to learn about" pointer.
- Cut: long expository paragraphs, "what carries forward" essays, route descriptions longer than one line, repeated cross-references, hedging. Keep: interactive figure links (only where the section survives), the one-line cross-refs that matter.
- Cross-references to other workbooks: refer by workbook and section *name* ("WB5 keystone", "WB4, Solomonoff"), not by § number, since every WB is being renumbered in parallel. Within a page, use anchors.
- Prose style: plain, physics-y, short sentences; no em-dash-heavy rhetoric; no "the result:" spoiler paragraphs full of undefined symbols. Don't add papers CRW hasn't been shown before except the keystones above and well-known classics (e.g. Belkin et al. 2019, Nakkiran et al. 2019, Nagarajan & Kolter 2019); verify any URL you add (fetch it).
- Target size: roughly a third (or less) of the current page's prose. WB2 can stay longer because of its exercises, but its prose should shrink just as hard.

## Mechanics

- Edit the builders (`notebooks/build_wbN.py`; WB2 also `wb2_s*.py` / `wb2_new_*.py`, which you may consolidate or rewrite). A pre-restructure snapshot of every builder and page is in `notebooks/_pre_trail_2026-10-09/`; do not modify it.
- Build with `cd notebooks && python build_wbN.py`; it must pass (balanced `<details>`, no TeX outside math, no dangling anchors). Update the builder's module docstring to describe the new shape.
- Do not edit other workbooks' builders, `nbkit.py`, `README.md`, or anything under `td/`; the coordinator handles those. If you need an nbkit change, say so in your report.
- Report back (concisely): the new section list with one line each; what you cut or demoted to references; any new URLs and whether you fetched them; an old→new anchor map for anchors that moved or vanished (`wbN-section-K` etc.); open questions for CRW.
