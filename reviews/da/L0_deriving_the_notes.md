# DA derivation deck (L0): build and verification log (2026-10-22)

**Request (CRW):** for the DA day (2026-10-23), CRW teaches a guided derivation of Louis Jaburi's notes rather than our prepared material. Wanted: slides that derive the notes step by step, as a reference/backup; Louis's notes linked and marked as the primary source on the website; "under construction" removed.

**Files**
- `da/lectures/L0_deriving_the_notes.html`: 91 slides, one derivation step each, in the notes' order (§1–§4.4; §5–6 of the notes are stubs). Same reveal.js/KaTeX stack and style as L1. Eyebrows name the item in the notes (PDF numbering). Notes' conventions (summed loss).
- `da/lectures/assets/L0/make_figures.py`: `unroll_to_if.png` (Ex. 4.2), `dog_cat.png` (Ex. 4.3; exact linear response by finite differences). Other figures reused from `assets/L1/`.
- `da/checks/L0_checks.py`: 65 numeric checks, all passing.
- `index.html`: DA section no longer "under construction"; CRW note naming Louis's notes as primary (web version with solutions, PDF without) and our material as backup; links to the notes and the deck. `da/README.md` likewise.
- Also: DA2 link label "TD §2.3" → "TD2.2 §3" (after the TD renumbering), rebuilt.

**Errata.** Where the notes are wrong the slide shows the corrected statement and its speaker notes begin "Correction:" (ledger E1–E13 in `DA_SPECS_v1.md`), per the spec's "correct but not pedantic".

**Review**
- Codex (high effort) adversarial math review of every slide and note: 22 findings (11 errors, mostly scope/assumption statements and two deck-introduced slips: the pull direction sign and the MAGIC adjoint recursion; plus the DLN decomposition's scope, Theorem 2's conclusion as set convergence, the 3.3(d) absolute-vs-relative cost, the ODE clock's 1/n). All 22 applied.
- Sonnet render check (headless Chromium, fragments forced visible, overflow measured per slide): 8 overflows fixed by layout-only changes; re-run after the math edits.

**Open for CRW**
- Training Dynamics still says "under construction" on the index; left as is pending CRW.
