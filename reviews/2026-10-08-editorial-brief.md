# Mysteries notebooks: instructor clarification, 2026-10-08

This records the instructor's direct clarification in the editing session. It supersedes conflicting agent-written workbook specs and handoff instructions. The original `Mysteries of Deep Learning` note is a sketch of interests, not a rigid specification.

- Students choose their own learning. There is no expectation that anyone finishes all the material. Preserve breadth and organize it so students can identify what interests them.
- Remove attributed instructor views; the instructor will add any personal positions himself.
- Use topic-appropriate heterogeneity. Derivations should mostly be exercises with worked answers. Link the original paper and provide enough setup, intermediate steps, hints, and solutions to learn without an LLM subscription.
- Reading-focused parts should offer only a few sentences describing what a reading shows, enough to decide whether to read it, plus the link. Do not replace papers with lengthy exposition.
  - This is a substantial change from spec v2's 'prose narrates ideas, not papers'. That older instruction is superseded. A 300-word historical essay followed by a reading blurb is still a violation; deleting only its heading or calling it 'short framing' does not satisfy the request. The paper is the reading, and the notebook helps students choose it. Substantial notebook text belongs to actual derivation setup, exercises, hints and worked solutions.
- Do not impose a universal claim/evidence/falsification rubric. Remove generic discussion padding, especially around derivations. Keep specific questions only where they genuinely help learn the material.
- Bayes/Gibbs/MLE distinctions may remain locally where useful; they are not a unifying theme across the notebooks.
- All solutions are available to all students. Students have full responsibility for how they use them.
- Keep plentiful material, including planning as inference, Solomonoff, perfect PRH, and subliminal learning if well organized. Breadth is not a defect merely because it exceeds one day.
- Remove shared schedules, convergence cards, and instructions to predict before opening figures. A meaningful local 'core' reading label is acceptable.
- Design and organization decisions and reviews must use Claude Opus 5.5 at high effort, as explicitly requested. Technical reviews may use other available agents.
- Use iterative critique and revision for each notebook, including independent sub-agent/sub-sub-agent checks. Treat agents kindly and respectfully. Do not mistake an earlier spec for instructor authority.

## Implementation boundaries

Edit builder sources, never hand-edit generated notebooks. Preserve the original topic note and historical specs. Do not change Training Dynamics materials. No training, paid compute, publishing, or external messages are needed.

Validate rebuilt exercise/solution pairs, local links and anchors, derivations and source claims as applicable. Record concrete findings, fixes, re-check outcomes and remaining uncertainties in this directory; do not claim source verification that was not performed.

Concurrent technical findings are recorded in `reviews/technical-sources.md` (WB1, WB3–5) and `reviews/wb2-technical.md`. Check these during revision and independent review; several concern errors in the original derivations that must not be copied into new exercises.
