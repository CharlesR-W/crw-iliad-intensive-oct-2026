# slt4_applications (NB4): presentation pass, 2026-10-14

Brief: `reviews/slt/2026-10-14-presentation-brief.md`. Builder: `slt/build_slt4_applications.py`.

## Changes

- Intro: one-line description of the page, notation line (tempered local posterior, SGLD, $\hat\lambda$), and a map figure of how each application hangs off NB1–NB3.
- §4.1: shorter pitch; schematic LLC-over-training figure (labelled "schematic, not data").
- §4.2: the shared identity stated once as a display, $\partial_h\mathbb E_h[O]|_0=-\beta\,\mathrm{Cov}_\beta(O,\Delta)$ under $\propto\varphi e^{-\beta(nL_n+h\Delta)}$, replacing an inaccurate pointer to "Notebook 1's dictionary" (NB1 does not define susceptibilities; it has the $\partial_\beta F$ identities in §1.1).
- §4.3: BIF as the same identity with $\Delta=\ell_i$.
- §4.4: code-length claim made self-contained ($F_n=-\log Z_n$ is the Bayes-mixture code length) instead of pointing to NB1 §1.3, which treats the Occam factor, not code length.
- Annotations: one sentence each, naming the NB1–NB3 idea used. No readings added or removed; ratings unchanged.

## Figures (`slt/assets/figures/make_slt4_applications_figures.py`, ~1 s)

- `slt4_applications_map.svg`: NB1 → NB2 → NB3 chain; arrows from NB3 to 4.1–4.4; dashed arrow from NB2 to 4.5.
- `slt4_applications_llc_curve.svg`: schematic loss and $\hat\lambda$ over training (log step), five stages, boundaries at the flat parts of $\hat\lambda$; $\hat\lambda$ rises in four stages and falls in one, loss falls in every stage. Shape follows the language-model run of Hoogland et al. (arXiv:2402.02364 Fig. 1: $\Delta\hat\lambda$ signs + + − + +, test loss falling every stage, boundaries at critical points of the smoothed LLC curve; checked via the arXiv HTML on 2026-10-14).

## Verification

- `slt/checks/slt4_applications_identities.py` (new; adapted from `_local_prior_2026-10-12/checks/`): finite-difference check of the §4.2 identity for a general $\Delta$, the susceptibility normalisation $\chi=(1/n\beta)\partial_h\mathbb E=-\mathrm{Cov}(O,L'-L)$, and the BIF case $\Delta=\ell_i$. All pass.
- `python checks/slt{1,2,3}_checks.py` still pass (not touched).
- Build, `--check`, `--run` pass (no code cells; 0 s).
- No URLs added. The spec's OpenReview link for the refined LLC still serves only a browser challenge to automated fetches (unchanged; arXiv 2410.02984 is given alongside).

## Critique (Codex, `--effort high`, read-only, one round)

Codex verified the §4.2 identity (sign, factor $\beta$), its §4.3 use, the three numerical checks, and the annotations against the abstracts; it found §4.5 neutral and no added readings. Changes made from its list:

1. §4.4: code-length claim restricted to $\beta=1$ (and fixed precision for continuous data).
2. §4.2: normalisation to Baker et al. stated ($\Delta=n\,\Delta L$, $\chi=(n\beta)^{-1}\partial_h\mathbb E=-\mathrm{Cov}(O,\Delta L)$); map labels now $-\mathrm{Cov}(O,\Delta L)$ and $-\beta\,\mathrm{Cov}(O,\ell_i)$.
3. §4.1: pitch no longer says the refined-LLC paper cuts stages; boundaries are "where the curve is flat", stages are not. Sketch's top axis relabelled "test loss" (Codex confirmed the sketch is qualitatively right against Hoogland et al. Fig. 1).
4. Map: NB2 box no longer shows $V(t)\sim ct^\lambda$ (which hid $m$); now "volume exponent $\lambda$; $\bar F_n\approx\lambda\log n+\dots$". Caption says 4.2/4.3 *extend* NB1 §1.1's identities.
5. Hinton & van Camp annotation now says what the paper does (Gaussian weight noise, error vs weight information).
- Dropped "with no evaluation" from the intro (restated framing).
- Not adopted: section pointers for each long reading (kept to one sentence per paper, as asked for this pass).

## Addenda applied (CRW, mid-run)

- "Read the blog posts first" block inserted after the subtitle, text copied verbatim from the block in `build_slt1_bayes_statmech.py` (identical to the brief's).
- Reused "Free energy crossover" figure from Liam Carroll, DSLT 4 (LessWrong, 2023), in §4.1, credited and linked in the caption; post URL fetched (200); image downloaded via LessWrong's GraphQL post HTML (cloudinary id `u9ngzvbeaj0svkukttg0`, as PNG), inspected; source URL recorded in the figure script docstring. DSLT 4's liquid/gas free-energy picture was considered and not used (physics illustration, not the posterior).

## Prose length

Strip-tags word count, excluding figures and the blog block: 1224 before, 1318 after (sections only: 987 → 1106). The page is slightly longer: the shared identity (one display) and its normalisation sentence, and Codex's correctness qualifications, outweigh the cuts.

## Streamlining pass (2026-10-14 afternoon; `reviews/slt/2026-10-14-streamline-brief.md`)

Counts (strip tags, figures removed, from the `<h1>` on; includes headings, paper titles and author lines; inline math = `math inline` spans on the page):

| | words | inline math | display |
|---|---|---|---|
| after morning pass | 1323 | 38 | 1 |
| after this pass | 898 | 11 | 1 |

Removed:
- §4.6 "Background for the whole day" (Hinton & van Camp, Watanabe 2022 survey, WBIC, Wei et al., Furman exercises) and the Watanabe books (`references(book(...))`); `book`/`references` imports dropped.
- "Read the blog posts first" block replaced by the "These notebooks are a backup" block, verbatim from the brief.
- Annotations cut to one line each; symbols in annotations replaced by words ("the LLC", "the posterior covariance of per-sample losses"); intro notation down to one symbol; captions cut to one sentence.
- §4.3 text no longer repeats the formula; it points to §4.2's identity in words. §4.5 pitch reduced to one sentence.
- No pointers to NB3 §3.2, effective theory or "CRW's argument" existed in NB4 (grep clean); NB4 references NB3 only as "Notebook 3". Map unchanged (it never showed §4.6).

Kept: all ten application papers with their ratings, the §4.2 identity (one display), the map, the DSLT 4 figure, the LLC sketch. `checks/slt4_applications_identities.py` unchanged and passing.

Critique (Codex, `--effort medium`, read-only): all ten annotations still match the abstracts. Applied: §4.4 code-length sentence regains "for continuous data, up to a fixed-precision term"; §4.2 display regains "∝", names β and φ, and says Baker et al. divide *the derivative* by nβ, leaving minus the covariance with the change in average loss; Chen annotation narrowed to "its polygon-shaped critical points"; two padding phrases deleted ("One line per paper, grouped by…", "Listed for completeness; not on the day's path").

Build, `--check`, `--run` (0 code cells, 0 s) pass.
