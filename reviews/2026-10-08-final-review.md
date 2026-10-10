# Mysteries notebooks: completed revision and review

The user's direct clarifications, recorded in [the editorial brief](2026-10-08-editorial-brief.md), governed this revision. The original sketch and historical agent-written specifications were preserved.

## Result

Six workbook pairs now offer section navigation, brief linked reading introductions, and derivation exercises with worked answers. Solutions are available to all students. Attributed instructor positions, agent-added personal essay readings, schedules, convergence cards, generic claim/evidence rubrics, and prediction-before-figure gates were removed. Bayes/Gibbs/MLE material remains where locally useful, without framing the whole day.

The personal essay links were removed under the default interpretation stated to the user during work; the user can add them back. WB6 §13 was removed with its navigation entry; no incoming Mysteries-workbook link targeted it.

## Iterative and nested review evidence

Design and organization used Claude Opus 5.5 with high effort through the Claude CLI. Each notebook had an owner revision, fresh independent Opus critique, corrections, and a targeted independent recheck. Critics were nested under notebook owners. Separate technical agents checked selected derivations and source claims, including nested technical checks for WB2 and WB5.

| Workbook | Editorial record | Additional technical evidence |
|---|---|---|
| 1 | [Owner, critique and recheck](wb1-editorial.md) | [Control-as-inference repairs](technical-sources.md) |
| 2 | [Initial design review](wb2-editorial.md), [revision and recheck](wb2-revision.md) | [Technical audit and numerical checks](wb2-technical.md) |
| 3 | [Owner, critique and recheck](wb3-editorial.md) | [Coding and RL scope repairs](technical-sources.md) |
| 4 | [Owner, critique, recheck and source links](wb4-editorial.md) | [Solomonoff normalization and scope](technical-sources.md) |
| 5 | [Owner, critique and recheck](wb5-editorial.md) | [Independent math check and dispositions](wb5-math-check.md) |
| 6 | [Owner, critique and recheck](wb6-editorial.md) | [Primary-source scopes](wb6-source-check.md), [exercise math and recheck](wb6-math-check.md) |

An initial coordinator hit the CLI's 600-second background-task ceiling, interrupting three owners. Those interrupted passes are not counted as completed loops. A replacement coordinator with that ceiling disabled completed the loops; its final statistics record 20 spawned and completed agents, depth two, with no failures or killed agents. WB2 ran in a separate completed Opus session. A fresh [format critic](reading-format-check.md) caught overlong essays in the first drafts; the final revisions addressed those findings.

Some final small wording fixes were checked by their owners or root rather than spawning a third critic; individual records identify these. Non-Opus helper calls concerned mechanical validation or technical algebra, not design decisions.

## Verification executed by root

- Regenerated all twelve notebooks from their six builders; no manual notebook edits.
- All six `--check` commands report fresh pairs, with matching shared content and the intended blank/filled answer cells.
- All workbook Python sources parse. Notebook JSON, unique cell IDs and anchors, local links, cross-notebook anchor targets, balanced hint blocks, and removed-framing scans pass.
- WB2 Part A numerical suite: **63/63 passed**. Part B: **ALL OK**. Both ran with single-threaded BLAS using the existing scientific Python environment. No mathematical changes followed those runs; a later cross-reference clarification does not affect them.
- The spectral figure's JavaScript passes `node --check`. Its caption and labels now distinguish deterministic-equivalent approximations from exact finite-sample risk.
- The shared validator was corrected to compare explicit variant placeholders without rejecting ordinary prose containing “Exercises” or “Solutions”, and to resolve local URL paths separately from fragments. A focused regression check passed.

The earlier owner reports say builds were not run because those CLI agents lacked a shell. The root checks above supersede those build-status limitations.

## Limits

This was an editorial revision with substantial independent mathematical and source checking, not a complete reproduction of every cited result. Some empirical details were checked only against abstracts, and some retained citations were not re-fetched; the notebook-specific records identify these. Browser rendering and every interactive control were not re-tested. No training or paid compute was run. Training Dynamics notebooks were outside this revision.

Start with the [notebook index](../notebooks/README.md).
