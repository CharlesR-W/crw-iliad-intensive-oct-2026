#!/usr/bin/env python3
"""Numerical checks for wb2_new_pacbayes.py (2.C.1 VC primer, 2.E PAC-Bayes).

Run: python checks/wb2_new_pacbayes_check.py   (numpy + scipy; well under a minute).
Each check prints PASS/FAIL.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import gammaln

rng = np.random.default_rng(0)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


# =========================================================== 2.C.1 VC primer
def labelings_thresholds(xs):
    xs = sorted(xs)
    ts = [xs[0] - 1] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1]
    return {tuple(int(x >= t) for x in xs) for t in ts}


def labelings_intervals(xs):
    xs = sorted(xs)
    cuts = [xs[0] - 1] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1]
    labs = {tuple(0 for _ in xs)}
    for a, b in itertools.combinations_with_replacement(cuts, 2):
        labs.add(tuple(int(a <= x <= b) for x in xs))
    return labs


for m in range(1, 9):
    xs = list(rng.uniform(size=m))
    pt, pi = len(labelings_thresholds(xs)), len(labelings_intervals(xs))
    ok = pt == m + 1 and pi == 1 + m + m * (m - 1) // 2
    ok &= pt == sum(math.comb(m, i) for i in range(2)) and pi == sum(math.comb(m, i) for i in range(3))
    check(f"growth functions m={m}: thresholds m+1, intervals 1+m+C(m,2) (= Sauer-Shelah, tight)", ok,
          f"{pt}, {pi}")
check("thresholds: VC=1 (1 point shattered, 2 not)",
      len(labelings_thresholds([0.3])) == 2 and len(labelings_thresholds([0.3, 0.6])) < 4)
check("intervals: VC=2 (2 points shattered, 3 not; (1,0,1) missing)",
      len(labelings_intervals([0.3, 0.6])) == 4 and (1, 0, 1) not in labelings_intervals([.1, .5, .9]))

# Sauer-Shelah second inequality
ok = all(sum(math.comb(m, i) for i in range(d + 1)) <= (math.e * m / d) ** d
         for d in range(1, 15) for m in range(d, 200))
check("sum_{i<=d} C(m,i) <= (e m/d)^d for 1<=d<=14, d<=m<200", ok)


def halfspace_realizable(X, y):
    """Is there (w,b) with w.x+b >= 1 on y=1 and <= -1 on y=0?  (LP feasibility; strict margin wlog)."""
    s = 2 * np.asarray(y) - 1
    A = -s[:, None] * np.hstack([X, np.ones((len(X), 1))])
    res = linprog(np.zeros(X.shape[1] + 1), A_ub=A, b_ub=-np.ones(len(X)), bounds=[(None, None)] * (X.shape[1] + 1))
    return res.status == 0


for d in (1, 2, 3, 4):
    X = np.vstack([np.zeros(d), np.eye(d)])
    ok = all(halfspace_realizable(X, y) for y in itertools.product([0, 1], repeat=d + 1))
    # explicit construction from the answer: b = s0/2, w_i = s_i - s0/2
    for y in itertools.product([0, 1], repeat=d + 1):
        s = 2 * np.array(y) - 1
        b = s[0] / 2
        w = s[1:] - s[0] / 2
        ok &= np.all((X @ w + b >= 0).astype(int) == np.array(y))
    check(f"halfspaces d={d}: {{0,e_1..e_d}} shattered (LP and explicit (w,b))", ok)
    worst = 0
    for trial in range(30):
        X2 = rng.normal(size=(d + 2, d))
        nreal = sum(halfspace_realizable(X2, y) for y in itertools.product([0, 1], repeat=d + 2))
        worst = max(worst, nreal)
        # Radon partition from the null vector of [X^T; 1^T]
        M = np.vstack([X2.T, np.ones(d + 2)])
        lam = np.linalg.svd(M)[2][-1]
        I, J = lam > 0, lam <= 0
        sI = lam[I].sum()
        p1, p2 = (lam[I] / sI) @ X2[I], (-lam[J] / sI) @ X2[J]
        yR = I.astype(int)
        if not (np.allclose(p1, p2) and not halfspace_realizable(X2, yR)):
            worst = 10 ** 9
    check(f"halfspaces d={d}: random d+2 points never shattered; Radon labelling unrealizable",
          worst < 2 ** (d + 2), f"max labellings {worst} < {2 ** (d + 2)}")

# sin(omega x) construction, exact arithmetic: sin(omega x_i) > 0 iff (omega x_i / pi) mod 2 in (0,1)
ok = True
for m in range(1, 11):
    xs = [Fraction(1, 2 ** i) for i in range(1, m + 1)]
    for y in itertools.product([0, 1], repeat=m):
        om_over_pi = 1 + sum((1 - y[j - 1]) * 2 ** j for j in range(1, m + 1))
        for xi, yi in zip(xs, y):
            r = (om_over_pi * xi) % 2
            pos = 0 < r < 1
            ok &= (pos == bool(yi)) and r not in (0, 1)
check("sin(omega x): x_i=2^-i, omega=pi(1+sum(1-y_j)2^j) realizes every labelling (exact, m<=10)", ok)
# and in floating point for small m
ok = True
for m in range(1, 8):
    xs = 2.0 ** -np.arange(1, m + 1)
    for y in itertools.product([0, 1], repeat=m):
        om = np.pi * (1 + sum((1 - y[j - 1]) * 2 ** j for j in range(1, m + 1)))
        ok &= np.all((np.sin(om * xs) > 0).astype(int) == np.array(y))
check("sin(omega x) construction in float64, m<=7", ok)


# =========================================================== 2.E.1 DV inequality
ok_ineq, ok_eq = True, True
for _ in range(200):
    K = rng.integers(2, 12)
    P = rng.dirichlet(np.ones(K))
    f = rng.normal(scale=3, size=K)
    rhs = math.log(P @ np.exp(f))
    for _ in range(20):
        Q = rng.dirichlet(np.ones(K) * rng.uniform(0.1, 3))
        ok_ineq &= Q @ f - np.sum(Q * np.log(Q / P)) <= rhs + 1e-12
    G = P * np.exp(f) / (P @ np.exp(f))
    ok_eq &= abs(G @ f - np.sum(G * np.log(G / P)) - rhs) < 1e-10
check("DV: E_Q f - KL(Q||P) <= ln E_P e^f on random discrete examples", ok_ineq)
check("DV: equality at the Gibbs measure dP_f/dP = e^f / E_P e^f", ok_eq)

# Occam special case: KL(delta_h || P) = ln 1/P(h)
P = rng.dirichlet(np.ones(7))
Q = np.zeros(7); Q[3] = 1
kl = sum(q * math.log(q / p) for q, p in zip(Q, P) if q > 0)
check("KL(point mass at h || P) = ln 1/P(h)", abs(kl - math.log(1 / P[3])) < 1e-12)

# Gibbs posterior minimizes E_Q r + KL/lambda, value -(1/lambda) ln E_P e^{-lambda r}
K = 9
P = rng.dirichlet(np.ones(K)); r = rng.uniform(size=K); lam = 7.0
Gq = P * np.exp(-lam * r); Gq /= Gq.sum()
obj = lambda Q: Q @ r + np.sum(Q * np.log(Q / P)) / lam
best_rand = min(obj(rng.dirichlet(np.ones(K))) for _ in range(5000))
check("Gibbs posterior minimizes E_Q r + KL/lambda with value -(1/lambda) ln E_P e^{-lambda r}",
      abs(obj(Gq) + math.log(P @ np.exp(-lam * r)) / lam) < 1e-12 and obj(Gq) <= best_rand)

# Hoeffding-lemma MGF bound for a Bernoulli(p) loss (worst case of [0,1]): exact MGF <= e^{lam^2/(8n)}
ok = True
for n in (1, 5, 50, 1000):
    for lam in (0.1, 1, 5, 30, 300):
        for p in np.linspace(0, 1, 41):
            t = lam / n
            log_mgf = n * math.log(p * math.exp(t * (p - 1)) + (1 - p) * math.exp(t * p))
            ok &= log_mgf <= lam ** 2 / (8 * n) + 1e-12
check("E_S exp(lam (R - R_hat)) <= exp(lam^2/(8n)) for Bernoulli losses (exact MGF)", ok)

# Maurer: xi(n) = sum_k C(n,k) (k/n)^k (1-k/n)^(n-k) <= 2 sqrt(n)
ok = True
for n in range(1, 3001):
    k = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
              + np.where(k > 0, k * np.log(k / n), 0) + np.where(k < n, (n - k) * np.log(1 - k / n), 0))
    xi = np.exp(lt).sum()
    ok &= xi <= 2 * math.sqrt(n)
check("Maurer's xi(n) <= 2 sqrt(n) for 1<=n<=3000", ok)

# Optimizing lambda: value sqrt(a/(2n)); with a ratio-2 grid, loss factor <= 3/(2 sqrt 2)
n, a = 1000, 37.0
f = lambda L: a / L + L / (8 * n)
Ls = math.sqrt(8 * n * a)
check("lambda* = sqrt(8 n a) gives a/lambda + lambda/(8n) = sqrt(a/(2n))",
      abs(f(Ls) - math.sqrt(a / (2 * n))) < 1e-12)
grid = 2.0 ** np.arange(0, 20)
worst = max(min(a_ / g + g / (8 * n) for g in grid) / math.sqrt(a_ / (2 * n)) for a_ in np.geomspace(0.01, 2 * n, 2000))
check("ratio-2 geometric grid loses at most factor 3/(2 sqrt 2) = 1.0607", worst <= 3 / (2 * math.sqrt(2)) + 1e-9,
      f"worst {worst:.4f}")

# Simulation: thresholds on [0,1] with label noise; uniform prior on a grid of 101 thresholds.
#   R(h_t) = eta + (1 - 2 eta)|t - t0| exactly.
eta, t0, Kh = 0.1, 0.3, 101
ts = np.linspace(0, 1, Kh)
Rtrue = eta + (1 - 2 * eta) * np.abs(ts - t0)
Pu = np.full(Kh, 1 / Kh)
delta = 0.1
for n in (20, 200):
    trials = 4000
    viol_fixed = {lam: 0 for lam in (5.0, 20.0, 80.0)}
    viol_mca = 0
    viol_occam_fixed = 0
    for _ in range(trials):
        x = rng.uniform(size=n)
        y = (x >= t0).astype(int) ^ (rng.uniform(size=n) < eta)
        Rhat = ((x[None, :] >= ts[:, None]).astype(int) != y[None, :]).mean(axis=1)
        for lam in viol_fixed:
            # sup over Q of [E_Q(R-Rhat) - KL/lam] = (1/lam) ln E_P e^{lam (R - Rhat)} (DV), so this is the exact
            # worst-Q violation event of the fixed-lambda bound
            if math.log(Pu @ np.exp(lam * (Rtrue - Rhat))) > math.log(1 / delta) + lam ** 2 / (8 * n):
                viol_fixed[lam] += 1
        # McAllester/Maurer bound, for a family of Q's: all point masses and Gibbs posteriors
        bad = False
        Qs = [np.eye(Kh)[i] for i in range(Kh)]
        for lam in (1, 3, 10, 30, 100, 300, 1000):
            g = Pu * np.exp(-lam * (Rhat - Rhat.min())); Qs.append(g / g.sum())
        for Q in Qs:
            nz = Q > 0
            KL = np.sum(Q[nz] * np.log(Q[nz] / Pu[nz]))
            if Q @ Rtrue > Q @ Rhat + math.sqrt((KL + math.log(2 * math.sqrt(n) / delta)) / (2 * n)):
                bad = True
        viol_mca += bad
    for lam, v in viol_fixed.items():
        check(f"fixed-lambda bound, n={n}, lambda={lam}: worst-Q violation rate <= delta={delta}",
              v / trials <= delta, f"rate {v / trials:.4f}")
    check(f"McAllester/Maurer bound, n={n}: violation rate (point masses + Gibbs Q's) <= delta",
          viol_mca / trials <= delta, f"rate {viol_mca / trials:.4f}")


# =========================================================== 2.E.2 Gaussian posteriors
def kl_gauss(mq, Sq, mp, Sp):
    p = len(mq)
    Spi = np.linalg.inv(Sp)
    d = mq - mp
    return 0.5 * (np.trace(Spi @ Sq) + d @ Spi @ d - p + np.linalg.slogdet(Sp)[1] - np.linalg.slogdet(Sq)[1])


def logpdf(x, m, S):
    p = len(m)
    d = x - m
    Si = np.linalg.inv(S)
    return -0.5 * (np.einsum("ij,jk,ik->i", d, Si, d) + p * math.log(2 * math.pi) + np.linalg.slogdet(S)[1])


p = 3
A = rng.normal(size=(p, p)); Sq = A @ A.T + 0.3 * np.eye(p)
B = rng.normal(size=(p, p)); Sp = B @ B.T + 0.5 * np.eye(p)
mq, mp = rng.normal(size=p), rng.normal(size=p)
xs = rng.multivariate_normal(mq, Sq, size=2_000_000)
mc = np.mean(logpdf(xs, mq, Sq) - logpdf(xs, mp, Sp))
check("Gaussian KL closed form vs Monte Carlo", abs(kl_gauss(mq, Sq, mp, Sp) - mc) < 5e-3,
      f"{kl_gauss(mq, Sq, mp, Sp):.5f} vs {mc:.5f}")

# isotropic prior N(w0, sP2 I), posterior N(w, Sigma): ||w-w0||^2/(2sP2) + 1/2 sum_k (s_k/sP2 - 1 - ln(s_k/sP2))
p = 6
w, w0, sP2 = rng.normal(size=p), rng.normal(size=p), 0.7
A = rng.normal(size=(p, p)); Sig = A @ A.T / p + 0.05 * np.eye(p)
sk = np.linalg.eigvalsh(Sig)
form = np.sum((w - w0) ** 2) / (2 * sP2) + 0.5 * np.sum(sk / sP2 - 1 - np.log(sk / sP2))
check("KL to isotropic prior: norm term + sum_k (s_k/sP2 - 1 - ln s_k/sP2)/2",
      abs(form - kl_gauss(w, Sig, w0, sP2 * np.eye(p))) < 1e-10)
sQ2 = 0.13
iso = np.sum((w - w0) ** 2) / (2 * sP2) + p / 2 * (sQ2 / sP2 - 1 + math.log(sP2 / sQ2))
check("KL isotropic posterior N(w, sQ2 I)", abs(iso - kl_gauss(w, sQ2 * np.eye(p), w0, sP2 * np.eye(p))) < 1e-10)

# E_Q Rhat = Rhat(w) + 1/2 tr(H Sigma) exactly for a quadratic Rhat, even with nonzero gradient
Hm = rng.normal(size=(p, p)); Hm = Hm @ Hm.T
gvec = rng.normal(size=p); c0 = 0.4
Rq = lambda v: c0 + (v - w) @ gvec + 0.5 * np.einsum("...i,ij,...j->...", v - w, Hm, v - w)
samp = rng.multivariate_normal(w, Sig, size=1_000_000)
check("E_Q Rhat = Rhat(w) + tr(H Sigma)/2 for quadratic Rhat (gradient term averages out)",
      abs(Rq(samp).mean() - (c0 + 0.5 * np.trace(Hm @ Sig))) < 2e-2 * np.trace(Hm @ Sig),
      f"{Rq(samp).mean():.4f} vs {c0 + 0.5 * np.trace(Hm @ Sig):.4f}")

# Optimal Sigma for the fixed-lambda bound; value (1/2lam) ln det(I + lam sP2 H)
lam = 40.0
Hpsd = Hm / 10


def obj_sigma(L_flat):
    L = np.tril(L_flat.reshape(p, p))
    S = L @ L.T + 1e-9 * np.eye(p)
    return 0.5 * np.trace(Hpsd @ S) + kl_gauss(w, S, w0, sP2 * np.eye(p)) / lam


L0 = np.linalg.cholesky(0.1 * np.eye(p)).ravel()
res = minimize(obj_sigma, L0, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
Sopt = np.linalg.inv(np.eye(p) / sP2 + lam * Hpsd)
val_closed = np.sum((w - w0) ** 2) / (2 * sP2 * lam) + np.linalg.slogdet(np.eye(p) + lam * sP2 * Hpsd)[1] / (2 * lam)
check("optimal Sigma_Q = (I/sP2 + lam H)^-1 and min value = norm/(2 sP2 lam) + ln det(I + lam sP2 H)/(2 lam)",
      abs(res.fun - val_closed) < 1e-7 and abs(obj_sigma(np.linalg.cholesky(Sopt).ravel()) - val_closed) < 1e-10,
      f"numeric {res.fun:.8f}, closed {val_closed:.8f}")

# Rescaling (w1, w2) -> (alpha w1, w2/alpha) at a minimum of a (w1 w2 - c)^2: norm and Hessian eigenvalue
a_, c_ = 1.3, 0.8
lam, sP2 = 50.0, 1.0
def bound_terms(alpha, w1=2.0):
    w2 = c_ / w1
    v = np.array([w2 / alpha, alpha * w1])
    H = 2 * a_ * np.outer(v, v)
    u0 = np.array([alpha * w1, w2 / alpha])
    if 0.2 < alpha < 5:  # central-difference check of H = 2a v v^T, v = (w2, w1) at the rescaled point
        Rf = lambda u: a_ * (u[0] * u[1] - c_) ** 2
        e, E = 1e-4, np.eye(2)
        Hn = np.array([[(Rf(u0 + e * (E[i] + E[j])) - Rf(u0 + e * (E[i] - E[j])) - Rf(u0 - e * (E[i] - E[j]))
                         + Rf(u0 - e * (E[i] + E[j]))) / (4 * e ** 2) for j in range(2)] for i in range(2)])
        assert np.allclose(H, Hn, rtol=1e-5, atol=1e-6), (H, Hn)
    return u0 @ u0 / (2 * sP2) + 0.5 * np.linalg.slogdet(np.eye(2) + lam * sP2 * H)[1]
alphas = np.geomspace(0.05, 20, 2001)
vals = np.array([bound_terms(al) for al in alphas])
al_bal = math.sqrt(abs((c_ / 2.0) / 2.0))  # alpha^2 = |w2/w1|
check("rescaling: complexity term minimized at the balanced representative |w1|=|w2|, unbounded away from it",
      abs(alphas[vals.argmin()] - al_bal) / al_bal < 0.01 and vals[0] > vals.min() + 5 and vals[-1] > vals.min() + 5,
      f"argmin alpha {alphas[vals.argmin()]:.3f} vs balanced {al_bal:.3f}")

# Random labels: KL >= 2n(0.9-0.1)^2 - ln(2 sqrt n / delta)
n, delta = 50_000, 0.05
lb = 2 * n * 0.8 ** 2 - math.log(2 * math.sqrt(n) / delta)
check("random labels: KL lower bound 2n(0.8)^2 - ln(2 sqrt(n)/delta) ~ 6.40e4 nats", abs(lb - 63990.9) < 1,
      f"{lb:.1f}; ln term {math.log(2 * math.sqrt(n) / delta):.2f}")
# Dziugaite & Roy random-label row: n=55000 binary, SNN train error 0.112 -> lower bound on KL from Pinsker form
n = 55_000
lb_dr = 2 * n * (0.5 - 0.112) ** 2 - math.log(2 * math.sqrt(n) / 0.035)
check("D&R random-label row: Pinsker-form lower bound on KL (~1.66e4) below their reported 201131",
      lb_dr < 201131, f"{lb_dr:.0f}")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
