# slt1_bayes — review and verification log

Builder: `slt/build_slt1_bayes.py`. Checks: `slt/checks/slt1_bayes_formulas.py` (about 3 min; prints `all checks passed`).

## Verification

### Formulas (in `checks/slt1_bayes_formulas.py`)

| Claim (section) | How checked |
|---|---|
| $F'(\beta)=\mathbb E_\beta[nL_n]$, $F''=-\mathrm{Var}_\beta[nL_n]$, $F(0)=0$, thermodynamic integration (§1.1, Ex 1.1b) | numerical, 1-D non-Gaussian posterior on a fine grid, finite differences |
| $\beta\mathbb E_\rho[nL_n]+\mathrm{KL}(\rho\|\varphi)=F+\mathrm{KL}(\rho\|p_\beta)\ge F$ (Ex 1.1b) | numerical, three Gaussian $\rho$, two $\beta$ |
| $F=-\log\sum e^{-F_j}$, $F_1-\log k\le F\le F_1$, $e^5\approx150$ (Ex 1.1a) | numerical (Riemann sums over a partition) |
| coverage $2\Phi(1.96/\sigma)-1$: 0.673 ($\sigma=2$), 0.99991 ($\sigma=\frac12$) (Ex 1.2a) | closed form plus simulation |
| information identity $C=H$ when realisable (Ex 1.2a) | sympy, Poisson log-rate model (general proof is in the solution) |
| $\bar Z_n=n\int e^{-nt}V\,dt=(2\pi/n)^{d/2}\det H^{-1/2}\varphi(w_0)=\Gamma(\frac d2+1)V(1/n)$, $\omega_d\Gamma(\frac d2+1)=\pi^{d/2}$ (§1.3, Ex 1.3a.4) | sympy, $d=1,2,3,5$ |
| Laplace error $O(1/n)$ (§1.3) | numerical, non-Gaussian 2-D integral: error ratio 3.94 between $n=50$ and $200$; also both notebook toys (figure) |
| coin: Stirling$(-\log Z_n)$ = Laplace exactly (Ex 1.3b) | sympy |
| BIC offsets quoted on the page: coin $-0.139$, logistic $-0.378$ nats (exact minus BIC limit) | closed form / quadrature at $w_0$ |
| $\mathbb E G_n=\mathbb E F_{n+1}-\mathbb E F_n-S$; $\mathbb E F_n-nS=\sum_{m<n}\mathbb E G_m$ (§1.4, Ex 1.4a) | exact sums over the binomial, coin with uniform prior (rule of succession) |
| $n\,\mathbb E G_n\to d/2$ (coin) | exact: 0.4935 at $n=200$, 0.4993 at $n=2000$ |
| $\mathbb E[nL_n(\hat w)]=nL(w_0)-\frac12\mathrm{tr}(H^{-1}C)+o(1)$ (Ex 1.4a.3) | Monte Carlo, exponential model on lognormal data (misspecified) |
| regular-case table: MLE $\nu$, Bayes $(\lambda-\nu)/\beta+\nu$, Gibbs $\lambda/\beta+\nu$, $\lambda=d/2$, $\nu=\frac12\mathrm{tr}(H^{-1}C)$ (§1.4, Ex 1.4b) | Monte Carlo with exact grid posteriors, misspecified exponential model ($\mathrm{tr}(H^{-1}C)=0.896$), $\beta=0.5,1,3$, $n=400$, 20000 samples: all within about 2% |
| same table, Poisson model with $D=\frac12,1,2$, $\beta=1,4$ | the notebook's own §1.4 code cell, $n=100$, 4000 samples: all within 2 s.e. |
| Ex 1.4c exact excesses $\sigma^2/2n$, $\frac12\log(1+\frac1n)$, $(1+\sigma^2)/2n$; gap $\frac12(\sigma^2-1-\log\sigma^2)$ | sympy, plus simulation for the Bayes term |

### Against Watanabe's results

- Survey arXiv:2211.10049 (fetched PDF): §2 defines $F_n$, $G_n$ and states $\mathbb E[G_n]=\mathbb E[F_{n+1}]-\mathbb E[F_n]$ (with $G_n$ including the entropy); §5.3 eq. (76) $\mathbb E[G_n]=L(\theta_0)+\lambda/n+o(1/n)$ at $\beta=1$; §3 property 1: regular gives $\lambda=d/2$, $m=1$. Our $G_n$ subtracts $S$; consistent.
- JMLR 2010 (watanabe10a, fetched PDF): Lemma 3, $\lim n\,\mathbb E[B_g(n)]=(\lambda-\nu)/\beta+\nu$; §4.3.1 and §5 (DIC part): regular unrealizable gives $\lambda=d/2$, $\nu=\frac12\mathrm{tr}(IJ^{-1})$. Our Bayes row agrees ($I=C$, $J=H$). The Gibbs row $(\lambda/\beta+\nu)/n$ is not stated in the fetched texts; derived in Ex 1.4b (Gibbs minus Bayes $=\frac12\mathbb E_X\mathrm{Var}_w[\log p]\to\nu/(n\beta)$, consistent with the survey's eqs. (64), (69)) and verified numerically above.
- Note: the spec's shorthand "Bayes ≈ d/2n, Gibbs ≈ d/n" is the well-specified $\beta=1$ case; under misspecification at $\beta=1$ Bayes stays $d/2n$ while Gibbs is $(d+\mathrm{tr}(H^{-1}C))/2n$ and MLE $\mathrm{tr}(H^{-1}C)/2n$.

### Readings and URLs (all fetched 2026-10-12)

- MacKay 1992: DOI resolves (to direct.mit.edu, which returns 403 to scripts); open-access PDF fetched from cs.uwaterloo.ca mirror and CaltechAUTHORS (200). Checked: §2 "The Evidence and the Occam Factor", eq. 2.5–2.6, Occam factor $=\Delta w/\Delta_0w$ "ratio of the posterior accessible volume ... to the prior accessible volume" (pp. 421–423).
- Hinton & van Camp 1993 (PDF fetched): §5 noisy weights, §5.2 bits back, $G(P,Q)=\mathrm{KL}$; §7.1 "A lemma from statistical mechanics" (free energy minimised by Boltzmann, equal to $-\log Z$).
- Watanabe 2022 survey: arXiv abs and PDF fetched; sections as above.
- Kleijn & van der Vaart 2012 (DOI → projecteuclid, fetched): abstract states posterior asymptotically normal under misspecification and credible sets not valid confidence sets. The covariance $(nH)^{-1}$ is our derivation (exact in Ex 1.2a's example); not checked against the paper's full text.
- Watanabe JMLR 2010: page and PDF fetched; Remark 1, Lemma 3, §4.3.1.
- Watanabe 2018 book: DOI resolves to Taylor & Francis; chapter 4 title "Regular Posterior Distribution" from the publisher listing (sections from a preview listing, not the book itself).

### Worksheet reuse

Ex 1.1a adapted from Iliad B.3 worksheet Exercises 1.7–1.8; Ex 1.2a part 1 extends Exercise 2.11(b). Exercise numbers checked against the compiled worksheet PDF.

## Critique rounds

### Round 1 (2026-10-12): Claude Opus design critic, Codex (GPT, effort high) technical critic

Design critic (12 points). Acted on:
- Too much for 90 min: cut Ex 1.3b (coin = Stirling; kept as a one-line remark under the §1.3 code) and Ex 1.4a part 3 (heuristic $\mathbb E[nL_n(\hat w)]$ route); shortened the §1.1 density-of-states paragraph; re-budgeted the route (§1.1 15 min) and said MacKay §§1–2 can replace §1.3's prose. Kept the Gaussian-location example threaded through Ex 1.2a and 1.4c (one example reused, not three), and the Poisson code as its numerical check.
- Dictionary: "ground-state energy" → reference energy (with a note that the ground state is at $\hat w$); "phase" → "macrostate"; "d usually small" → "d fixed as n grows"; no-thermodynamic-limit note rewritten (switches sharpen as n grows, as in any finite system cooled to zero temperature, but no thermodynamic-limit phase transition); column renamed "Note".
- §1.4 differencing step now labelled heuristic in the main text.
- "Regular case" defined once in §1.2.
- Notation: $S_{\rm thermo}$ in Ex 1.1b solution; Watanabe annotation restated with our $G_n$ (subtracting $S$); "$\mathbb E$ with no subscript averages over $D_n$" added to §1.4 notation.
- Readings: Kleijn–van der Vaart moved to the final extras line; pointer sentence deleted; Hinton–van Camp moved after Ex 1.1b, importance 3 → 2; Watanabe 2022 length 3 → 2, "founder of the subject" → "Watanabe's own survey".
- §1.3 pitch now names the volume point; core exercise list now includes 1.4a; setup cell introduced; duplicate inline hint in Ex 1.2a removed; BIC "dominates comparisons" → "decides between models ... that fit about equally well"; Poisson prior offset explained; removed the λ/ν restatement paragraph from §1.4 text.
- Not acted on: (12) the prior ↔ "density of states" mapping. Kept the spec's mapping in the table with "(strictly, the reference measure ...)" and a two-line note; flagged as a question for CRW.

Codex (9 definite, 6 minor). Acted on:
- (1) MLE expected log loss literally infinite at finite n (all-zero samples): added one line in §1.4 that "expected excess" means the mean of the limiting distribution, with the coin example. Did not change the Poisson code (all-zero probability below $2^{-400}$).
- (2) Conditions for BvM/Laplace: added posterior concentration / no distant near-optimal region with appreciable prior mass (automatic for compact $W$) in §1.2, and a gap condition in Ex 1.3a; $\beta>0$ fixed as $n\to\infty$ added to the notation table.
- (3) Ex 1.4c exact finite-n threshold: now $\sigma^2<n\log(1+1/n)$ ($\log2$ at $n=1$, rising to 1), plus the large-n statement. Checked.
- (4) Cumulative loss: kept the $n(L(w_0)-S)$ term and attributed the $O(1)$ to Laplace, not to summing remainders.
- (5) ground state: fixed (above).
- (6) TV omitted Gaussian mass outside [0,1]: fixed in code; $n=10$ value is now 0.127, matching Codex's independent 0.126925 (added to checks).
- (7) "misspecified ⇒ C ≠ H": now "C need not equal H ... can lose coverage".
- (8) Watanabe annotation $G_n$ convention: fixed.
- (9) λ/n "general form": now "under his assumptions".
- Minor: (10) Ex 1.4b posterior-mean shift explained (pointwise $O(1/n)$, averages to $O(n^{-3/2})$ because $\mathbb E_q g=O(n^{-1/2})$); (11) moot (exercise cut); (12) volume solution uses ≈; (13) as dictionary fix above; (14) BIC "for large n" above the exact curve; (15) Gibbs score described as an average of KL divergences.
