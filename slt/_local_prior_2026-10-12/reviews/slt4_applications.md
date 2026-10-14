# slt4_applications (NB4, "Applications"): review log

Builder: `slt/build_slt4_applications.py`. Output: `slt/slt4_applications.{html,ipynb}`, 7 cells, no code cells (`--run` trivially 0 s).
Checks: `slt/checks/slt4_applications_identities.py`.

## Formula checks

The script `checks/slt4_applications_identities.py` (all pass) covers only the first four items below; the rest are read off the papers, not re-derived or computed.

- Susceptibility: for p_h ∝ φ e^{−nβ L_h}, L_h = (1−h)L + hL′, the finite-difference (1/nβ)∂_h⟨ϕ⟩ at h=0 equals −Cov_β(ϕ, L′−L) on a 1-D grid (agreement to 1e-8).
- BIF: for p ∝ φ exp(−Σβ_i ℓ_i), finite-difference ∂⟨ϕ⟩/∂β_i at β=1 equals −Cov(ℓ_i, ϕ) on a 2-D grid regression posterior.
- Loss kernel: Cov(ℓ_i, ℓ_j) matrix is PSD (min eigenvalue ~1e-19) and K_ij = −BIF(z_i, ℓ_j).
- Regular volume law: K = |w|² in d = 3, uniform prior, Monte Carlo V(t) slope 1.49 vs d/2 = 1.5 (the regular case λ = d/2 used in §4.4).
- MDL redundancy formula and its conditions: read off Theorem 1 and the text above it in arXiv:2510.12077 (realisable q ∈ M, distributions on a finite set X bounded below by a constant; redundancy λ log n − (m−1) log log n + O_p(1)). Not re-derived.
- WBIC: read off Theorem 4 of the JMLR paper (PDF fetched): WBIC = nL_n(w_0) + λ log n + U_n √(λ log n / 2) + O_p(1), "first two main terms are equal to those of F".
- Local free energy F_n(W_α) ≈ nL_α + λ_α log n: as in Chen et al. §4.1 eq. (6)–(7).

## URLs fetched (2026-10-12)

All arXiv abs pages returned 200; metadata and abstracts via the arXiv export API; PDFs downloaded and section headings extracted with pdftotext.

| Item | URL | Status / what was checked |
|---|---|---|
| Chen et al. | arxiv.org/abs/2310.06301 | abstract, §1, §4–5 headings, §4.1 text |
| Hoogland et al. | arxiv.org/abs/2402.02364 | abstract; PDF header "To appear in TMLR (07/2025)"; Fig. 1, §4–7; two-layer attention-only LM + ICL regression |
| Refined LLC | arxiv.org/abs/2410.02984 | abstract, §3.2 definitions. **Spec URL openreview.net/forum?id=SUc1UOWndp could not be fetched** (OpenReview browser challenge, 403 on API and PDF); a web search suggests the forum id may be `SUc1U0Wndp` (zero, not O). Replaced with the arXiv abs link. ICLR 2025 venue confirmed in the ICLR 2025 proceedings listing (proceedings.iclr.cc/paper_files/paper/2025, hash fad7c708…). |
| Baker et al. | arxiv.org/abs/2504.18274; page links the ICLR 2026 proceedings page (fetched, 200), whose author list adds Pathak | abstract; PDF header "Published as a conference paper at ICLR 2026"; Def. 2.1, Lemma 2.2 (χ = −Cov_β(ϕ, ΔL)), eq. (3) mixture |
| Gordon et al. | arxiv.org/abs/2601.12703 | abstract (510 clusters, 50% match SAE features, modes decomposition) |
| Kreer et al. | arxiv.org/abs/2509.26544 | abstract; arXiv comment "ICLR 2026 camera-ready"; §2.2 eq. (3)–(4), §2.3 localisation, App. A.1–A.2 |
| Loss kernel | arxiv.org/abs/2509.26537 | abstract; §2.2 probe distribution, §2.3 definition |
| Urdshals et al. | arxiv.org/abs/2510.12077 | abstract; §3 Theorem 1 and assumptions |
| Clift et al. | arxiv.org/abs/2103.16080 | abstract, section list (§6.1 programs as phases, §7 experiments) |
| Murfet & Troiani | arxiv.org/abs/2504.08075 | abstract, table of contents |
| Hinton & van Camp | cs.toronto.edu/~hinton/absps/colt93.pdf | PDF, abstract (9 pages) |
| Watanabe survey | arxiv.org/abs/2211.10049 | abstract, section list; journal from worksheet bib (Information Geometry 7, 2024) |
| WBIC | jmlr.org/papers/v14/watanabe13a.html | page and PDF; Theorem 4 |
| Wei et al. | arxiv.org/abs/2010.11560 | abstract, journal ref (TNNLS 34(12) 2023); published author order (Wei first) via Crossref, DOI 10.1109/tnnls.2022.3167409 |
| Furman exercises | lesswrong.com/posts/3HYqTAi4kD35G3BzQ/… | page text: author, date 30 Aug 2024, exercise topics |
| Watanabe 2009 | doi.org/10.1017/CBO9780511800474 | resolves to the CUP book page |
| Watanabe 2018 | doi.org/10.1201/9781315373010 | resolves to the Taylor & Francis book page |
| (cut) Carroll et al. | arxiv.org/abs/2501.17745 | abstract; was a Claude-suggested extra, cut in round 1 |
| (cut) Lee et al. | arxiv.org/abs/2510.12071 | abstract; was a Claude-suggested extra, cut in round 1 |

## Corrections to the spec's references (silent in the page)

- Refined LLC: OpenReview link replaced by arXiv 2410.02984 (see above). Authors are Wang, Hoogland, van Wingerden, Furman, Murfet.
- Wei et al.: the arXiv listing puts Murfet first; the published TNNLS version puts Wei first, matching the spec's "Wei et al."; the page uses the published order.
- Hoogland et al.: no TMLR link was in the spec; the arXiv abs page (v3, "to appear in TMLR") is used.

## Not verified

- The OpenReview forum id in the spec (no access to OpenReview).
- Section-level content of the long papers beyond abstracts, the cited section headings and the specific passages listed above.

- Baker et al.: arXiv lists four authors; the ICLR 2026 version adds Vinayak Pathak. The page uses the ICLR author list and links the proceedings page, with the arXiv id alongside.

## Critique round 1 (one round, as allowed for NB4)

Design critic: Claude Opus subagent. Technical critic: Codex (GPT, `--effort high`, read-only).

Changes made:
- Notation: observable renamed $O(w)$ (was $\phi$, clashing with the prior $\varphi$); loss kernel written $\mathcal K$; $K(w)$ defined in §4.4. (design 1)
- §4.1: question split into the posterior-as-$n$-grows question and the separate checkpoint-LLC question; local free energy defined; "can take over" instead of a categorical switch; one-line note that "phase transition" is a smooth crossover at finite $d$ with no thermodynamic limit; SGLD spelled out. (design 2, 3, 6; Codex 11)
- §4.2: posterior now defined with the population loss $L_h$, as in Baker et al. eq. (1)–(2), so the mixture and the posterior are consistent; exactness attributed to differentiating under the integral, not to linearity in $h$. (Codex 1, 9; design 6, 7)
- §4.3: $\mathcal K=-\mathrm{BIF}(z_i,\ell_j)$ written explicitly; per-sample $\beta_i$ tied to $nL_n=\sum\ell_i$. (Codex 10)
- §4.4: regular case stated for realisable models with $O_p(1)$ remainder; the theorem's App. E.1 analytic conditions mentioned; volume is of $\{K<t\}$ as in the paper. (Codex 2, 3)
- WBIC annotation: "under conditions that cover singular and some unrealisable models" (Watanabe's Fundamental Condition (4) can fail when singular and unrealisable). (Codex 4)
- Kreer annotation: "under a Laplace approximation". (Codex 5)
- Baker author list per ICLR 2026. (Codex 6)
- Pitch rewritten to say who wants the page, that it has two identities and no code/exercises, and that nothing depends on it; route numbered 4.1–4.5. (design 4, 5)
- Trimmed jargon in annotations (Chen, Baker, Murfet & Troiani). (design 8)
- Wei et al. framed as an argument; importance 4 → 3. Ratings: 5 only for the keystone, 4 for other section leads. (design 9, 10)
- Readings for the whole day tagged by notebook and ordered NB1 → NB3. (design 11)
- Claude-suggested extras removed (Lee et al. cut on the coordinator's instruction; Carroll et al. cut as not clearly earning its place, since §4.1 already has Chen et al. and Hoogland et al.). (design 12)

Not adopted: design 7's "because $h$ enters linearly" (Codex 9 is right that linearity is not needed); design 13's sibling links (no sibling builders exist yet, and the kit fails on links to missing siblings; plain-text "NB2 §3" references used instead). Codex 8 (checker coverage) addressed by narrowing the claim above rather than adding checks.
