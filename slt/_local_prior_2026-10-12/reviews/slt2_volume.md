# slt2_volume — review log

Writer 2, 2026-10-12. Builder `slt/build_slt2_volume.py`; checks `slt/checks/slt2_volume_checks.py`.

## Verification

### Formulas (script: `slt/checks/slt2_volume_checks.py`, all checks pass, ~1 min CPU)

Symbolic (sympy) or high-precision numerical (mpmath / scipy) checks of:

- §1 volumes for the uniform prior: `w^{2k}` (V = t^{1/2k}), `x^2+y^4` (V = 0.874 t^{3/4}), `(ab)^2` (V = sqrt t (1 - log t/2), v = t^{-1/2}(-log t)/4).
- Eq. (1) (Zbar_n = n ∫ e^{-nt} V dt), numerically on `(ab)^2`.
- Regular-case constant c Γ(d/2+1) = φ(w0)(2π)^{d/2} det H^{-1/2} for d = 1,2,3,5, plus a direct 2-d Laplace check.
- §2 ζ for w^2, w^4, (ab)^2, a^{2k1} b^{2k2}; ζ = ∫ t^z v dt for (ab)^2; the dictionary integrals (pole order m ↔ (-log t)^{m-1}/(m-1)!) for m = 1,2,3; ∫ e^{-nt} t^{λ-1}(-log t) = Γ(λ) n^{-λ}(log n - ψ(λ)); ψ(1/2) = -γ - 2 log 2.
- Mellin-in-n identity ∫ n^{-z-1} Zbar_n dn = Γ(-z) ζ(z) on w^2 and (ab)^2 at two z values.
- (ab)^2 closed form (√π/4) n^{-1/2}(log n + γ + 2 log 2) against quadrature at n = 10 … 1e6 (rel. err ≤ 2e-7, the n = 10 deviation being the e^{-n} term), and the local-slope values 0.348 / 0.411 / 0.437.
- a^2 b^4 two-pole expansion ½Γ(¼)n^{-1/4} − ½Γ(½)n^{-1/2} at n = 1e2, 1e4.
- Gaussian prior on w^2: ζ = 2^z Γ(z+½)/√π, Zbar = (1+2n)^{-1/2}, residues at z = −½−k equal the binomial terms for k = 0…4 (sympy residue).
- Cubic: ζ = 2^{-z}/(6z+1) (uniform prior), same ζ in θ = μ^3 coordinates with push-forward prior (1/6)|θ|^{-2/3}, K''(μ0) = 9μ0^4, Gaussian-prior ζ = 4^z Γ(3z+½)/√π.
- §3: worksheet n* = 571.3 (and the spurious root 1.011), width 2/f' ≈ 12; iteration n ← r log n; n* ~ r(log r + log log r) error shrinking with r; d = 8 example exact n* 1478 vs asymptotic-with-constants 1478 vs leading 1457; dFbar/dn = E_post[K]; E_post[nK] → λ for w^2, w^4; width formula 0.159.
- λ ≤ d/2: proof is the worksheet's (logic checked by hand); spot checks on the examples.

### Code
`python build_slt2_volume.py`, `--check`, `--run` all pass; `--run` total ≈ 11 s, slowest cell 4.6 s.

### Sources and URLs (all fetched 2026-10-12)
- Keystone, Hoogland & van Wingerden LessWrong post: fetched via the LessWrong GraphQL API (full markdown). Section "From Basins to Model Complexity" contains V(ε) ~ cε^λ, the w1^4+w2^4 (λ=1/2) and w1^2 w2^4 (λ=1/4) examples, the free energy formula and the two-basin "receding sea" cartoon. The post says "volume scaling is invariant under reparametrizations"; the annotation qualifies this to smooth invertible ones.
- Watanabe 2022 survey, arXiv:2211.10049: PDF text extracted; §4.1 defines ζ and the state density v, states ζ = Mellin transform of v and (z+λ)^{-m} ↔ c0 t^{λ-1}(−log t)^{m-1}; §5.2 states F_n = nL_n(θ0) + λ log n − (m−1) log log n + χ(ξ_n) + o_p(1) and the expectation version.
- Furman, "Singular learning theory: exercises" (LessWrong): fetched via GraphQL; problems 2, 5, 12, 13, 17 as described in the annotation.
- Flajolet, Gourdon & Dumas 1995: PDF fetched; §1 basic properties, §2 "the fundamental correspondence" with Theorems 3 (direct mapping) and 4 (converse mapping).
- Book URLs (Cambridge 2009, Routledge 2018): HTTP 200.
- Iliad B.3 worksheet: Ex 3.4, 3.5, 3.6, 3.7, 4.1, 4.2 adapted (numbering checked against the compiled PDF).

### Not verified directly
- Watanabe's free energy theorem conditions are paraphrased from the survey and the worksheet (books not consulted).
- Meromorphic continuation with negative rational poles attributed to Atiyah (1970) and Bernstein–Gel'fand (1969): confirmed via web search (arXiv:1106.0482, arXiv:2110.10299 cite both); original papers not opened.

## Critique rounds
