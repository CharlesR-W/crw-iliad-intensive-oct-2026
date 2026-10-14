# slt3_temperature — review and verification log

Builder: `slt/build_slt3_temperature.py`. Checks: `slt/checks/slt3_temperature_formulas.py` (about 5 s; prints `ALL PASS`).
Run: `python build_slt3_temperature.py --run` (from `slt/`), about 55 s on CPU, slowest cell about 20 s.

## Verification

### Formulas (in `checks/slt3_temperature_formulas.py`)

| Claim (where) | How checked |
|---|---|
| $K=w^4+\delta w^2$: $\lambda=\frac14$ at $\delta=0$, $\frac12$ for $\delta>0$ (§3.1) | sympy: exact sublevel half-width, limit of $V(t)/\sqrt t$ |
| $\lambda_{\rm eff}=-d\log\bar Z_n/d\log n=n\,\mathbb E_n[K]$ (Ex 3.1b.1) | numerical: finite difference of $\log\bar Z_n$ vs quadrature of $n\langle K\rangle$, two $\delta$, three $n$ |
| Flat prior: $\lambda_{\rm eff}=\frac14+\frac s2\langle u^2\rangle_s$, $s=\delta\sqrt n$; depends only on $n\delta^2$; $0.367$ at $n\delta^2=1$; limits $\frac14,\frac12$ (Ex 3.1c) | numerical: direct quadrature at $(n,\delta)=(10^4,10^{-2})$ and $(10^6,10^{-3})$ vs the scaling form; integration-by-parts identity $4\langle u^4\rangle+2s\langle u^2\rangle=1$; closed form $g(s)=\frac{\sqrt s}2e^{s^2/8}K_{1/4}(s^2/8)$ (not used in the text) |
| $K=w^4+2aw^2$ for the model $y=w^2x$, $\sigma^2=\frac12$, truth $y=-ax+\varepsilon$; $\mathrm{sd}(\delta_n)=\sqrt{2/n}$ (Ex 3.1a) | sympy expansion; Monte Carlo over 4000 data sets at $n=1000$ |
| $\lambda_{\rm eff}=\lambda-(m-1)/\log n+o(1/\log n)$; $K=(ab)^2$ on $[-1,1]^2$ gives $0.45$ at $n=10^8$ (Ex 3.1b.3) | exact 1-D reduction $\bar Z_n=2\sqrt{\pi/n}\,G(\sqrt n)$, $G(X)=\int_0^X\mathrm{erf}(v)/v\,dv$; checked against 2-D quadrature at $n=10^4,10^8$; $(\log n)(\lambda-\lambda_{\rm eff})\to1$ |
| WBIC leading order: $\beta=1/\log n$; estimator algebra (Ex 3.3a) | sympy |
| Regular case: $\hat\lambda=\frac d2-\frac{n\beta}2\hat w^\top J\hat w$, $\mathbb E\hat\lambda=\frac d2(1-\beta)$ (Ex 3.3b.2) | Monte Carlo, $d=3$ |
| $\gamma$ cutoff: $\hat\lambda=\frac12/(1+\gamma/(n\beta h))$ (Ex 3.3c) | sympy |
| Aoyagi–Watanabe (2005) reduced-rank-regression $\lambda$, $m$ (§3.3 text and code) | reproduces all nine entries and multiplicities of Drton & Plummer (2017, arXiv:1309.0911) Table 1 ($N=5$, $M=3$), whose values are quoted from Aoyagi–Watanabe; small cases checked by hand ($M=N=H=1$, $r=0,1$; $M=N=2$, $H=1$, $r=0$); identical to the worksheet's statement (main.tex, Theorem `thm:aoyagi-watanabe`) |

### Code checks (in the notebook)

- SGLD $\hat\lambda$ on $K_n=m_2w^4+\delta_nw^2$ ($n=10^6$, $\delta=10^{-2}$) vs exact $\lambda_{\rm eff}(n\beta)$ by quadrature: agree to about $0.01$ over $n\beta\in[10^2,10^6]$.
- Normal crossings: SGLD on the population $K$ agrees with quadrature $\lambda_{\rm eff}(n\beta^*)$ to $\le0.025$; the data effect is shown separately over 6 data sets.
- DLN ($M=3,N=5,H=3$, $r=0..3$): $\hat\lambda$ within 3–16% below the Aoyagi–Watanabe $\lambda$; checked during tuning that full-batch runs with smaller $\epsilon$ give the regular-case prediction $\frac{15}2(1-\beta)$ at $r=3$ (6.70 vs 6.68 at $n=10^4$).

### Papers and URLs (fetched 2026-10-12)

| Claim | Source checked |
|---|---|
| Localised posterior, $\beta^*=1/\log n$, estimator (Def. 2), SGLD update (§4.4), DLN check up to $10^8$ params with Aoyagi (2024) values (§5.1), App. D estimator-of-localised-free-energy + conjecture, App. H advice (MALA acceptance 0.9–0.95, flat loss trace, $\gamma$ 1–10) | arXiv HTML of 2308.12108 (v2); AISTATS 2025 = PMLR v258 lau25a (page fetched) |
| WBIC definition; Theorem 3 (unique $\beta^*\in(0,1)$); Theorem 4 statement incl. $\beta=\beta_0/\log n$, $U_n$ mean zero, Gaussian limit, $\mathbb E U_n^2<1$ if realisable; conditions (compact $W$, analytic, condition (12) holds if realisable or regular) | JMLR PDF, extracted text |
| SGLD origin | Welling & Teh, ICML 2011 PDF (title, venue) |
| Furman notebook, LessWrong post (Hoogland & van Wingerden, 2023-10-11), Watanabe books (CUP 2009 doi; CRC 2018 doi) | fetched, titles confirmed |
| Aoyagi & Watanabe 2005 | doi resolves (Elsevier, paywalled); formula verified via Drton–Plummer Table 1 instead |

### Not verified

- Aoyagi & Watanabe (2005) itself (paywalled); the formula is verified only through Drton & Plummer's tabulated values for one $(M,N)$ and small hand cases.
- That the local learning coefficient at the specific centre used in the DLN cell equals the global Aoyagi–Watanabe $\lambda$; the text says only that Lau et al. compare with these values, and does the same.

## Critique rounds

### Round 1 (design: Claude opus subagent; technical: Codex, effort high)

Design critic, adopted:
- Cut the "loss scale 1/(nβ)" sentence and the closing "not a failed estimate / reports the exponent at a different scale" lines from the hyperparameter paragraph, and the "β only selects the sample size" phrase in Ex 3.3b: too close to CRW's §3.2 argument. Kept only the exact statement (β-dependence = n-dependence of λ_eff in the idealisation), which the spec's "hyperparameter dependence read as scale dependence" asks for.
- §3.3 notation now defines λ, m, λ_eff, w0, K; J and C defined in Ex 3.3b; the exact free-energy temperature renamed β_F (β* is 1/log n throughout); batch size renamed b (m is multiplicity); note that H is a width in the DLN.
- Removed `checks/` paths and the Drton–Plummer link from student text (kept in this log); the Aoyagi–Watanabe case list replaced by "implemented in `aoyagi_watanabe`" plus the four values used.
- Watanabe paragraph: shorter conditions parenthesis. Ex 3.3a.4: added the variance step. Ex 3.1a.4: added the payoff. Opening: clearer first sentence and a prerequisite line. Removed duplicate statements of the crossover. SGLD "temperature" sentence clarified. Normal-crossing reading split into the three effects, one line each. Readings paragraph shortened. §3.2 question made neutral.
- Not adopted: the claim that the .ipynb ships the full SGLD solution (checked: it ships the stub; added one line saying later cells need it and the solution is on the page). Also not adopted: the suggestion that the double well is "invisible" to the posterior (it changes nK_n by O(1) at the posterior's scale).

Codex, adopted (all ten):
1. "In any family exact degeneracy is measure zero" was false in general (e.g. symmetries); restricted to the family K_δ.
2. "O(ε) bias" now qualified (well-behaved U; steep w^4 can diverge at too large a step).
3. Term-by-term differentiation is now an explicit assumption (true for power–log expansions), not justified by "Laplace transform".
4. Flat-prior assumption stated at the first scaling claim; Ex 3.1c uses a flat prior on R; the post-plot sentence now quotes the computed range (0.24–0.30 for n = 10^2..10^7 at δ = 10^-4) instead of "what every sample sees".
5. "Random term of size up to √(λβ/2)" → O_p(√β), RMS below √(λβ/2) in realisable models.
6. Last column of the normal-crossing table now labelled as including Monte Carlo error.
7. Ex 3.3e answer: added the small shifts from m2 ≠ 1, δ_n ≠ δ.
8. Theorem 3 uniqueness: added "when L_n is not constant".
9. γ: "do not count" → suppressed / partly counted.
10. "log–log plot" → plot of F̄_n against log n.
