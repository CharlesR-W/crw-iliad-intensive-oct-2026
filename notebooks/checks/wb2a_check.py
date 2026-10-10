#!/usr/bin/env python3
"""Numerical checks for every nontrivial formula in WB2 sections 1-5 (wb2_cells_a.py).

Run: python checks/wb2a_check.py   (numpy only; ~1 minute).  Each check prints
the formula, the closed form, a Monte Carlo or brute-force value, and PASS/FAIL.
"""
import itertools
import math

import numpy as np

rng = np.random.default_rng(0)
results = []


def report_abs(name, err, atol):
    ok = abs(err) <= atol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: |error|={abs(err):.3g} (atol {atol})")


def report(name, closed, numeric, rtol):
    ok = abs(closed - numeric) <= rtol * max(abs(closed), 1e-12)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: closed={closed:.6g} numeric={numeric:.6g} (rtol {rtol})")


# ---------- §1.A: memorizer example and E[excess] <= E[gap] for ERM ----------
# Constant predictors on y ~ N(mu, s2): excess = (ybar-mu)^2, E = s2/n; gap E = 2 s2/n.
n, s2, mu, T = 10, 1.3, 0.7, 400_000
Y = mu + math.sqrt(s2) * rng.standard_normal((T, n))
yb = Y.mean(1)
excess = (yb - mu) ** 2
train = ((Y - yb[:, None]) ** 2).mean(1)
gap = s2 + excess - train
report("1.A mean-estimation E[excess]=s2/n", s2 / n, excess.mean(), 0.01)
report("1.A mean-estimation E[gap]=2 s2/n", 2 * s2 / n, gap.mean(), 0.02)

# ---------- §1.B: OLS ----------
n, p, s2 = 30, 8, 0.8
# Wishart inverse mean: E[(X^T X)^{-1}] = I/(n-p-1) for Gaussian rows
T = 100_000
X = rng.standard_normal((T, n, p))
W = np.einsum("tni,tnj->tij", X, X)
Winv = np.linalg.inv(W)
tr = np.trace(Winv, axis1=1, axis2=2)
report("1.B E Tr (X^T X)^{-1} = p/(n-p-1)", p / (n - p - 1), tr.mean(), 0.01)
report("1.B E[(W^{-1})_11] = 1/(n-p-1) (inverse chi^2_{n-p+1})", 1 / (n - p - 1), Winv[:, 0, 0].mean(), 0.01)
# Fixed design: OLS mean/cov, excess = sigma^2 Tr (X^T X)^{-1}
Xf = rng.standard_normal((n, p))
beta = rng.standard_normal(p)
T = 200_000
E = math.sqrt(s2) * rng.standard_normal((T, n))
Yf = Xf @ beta + E
B = np.linalg.solve(Xf.T @ Xf, Xf.T @ Yf.T).T
report("1.B E||b-beta||^2 = s2 Tr(X^T X)^{-1} (fixed X)", s2 * np.trace(np.linalg.inv(Xf.T @ Xf)),
       ((B - beta) ** 2).sum(1).mean(), 0.01)
cov = np.cov(B.T)
report_abs("1.B Cov b = s2 (X^T X)^{-1} (max entry rel. err)", np.abs(cov - s2 * np.linalg.inv(Xf.T @ Xf)).max() / np.abs(s2 * np.linalg.inv(Xf.T @ Xf)).max(), 0.02)
# Random design: E excess = s2 p/(n-p-1), E train = s2 (n-p)/n, gap = s2[(n-1)/(n-p-1) - (n-p)/n]
T = 100_000
X = rng.standard_normal((T, n, p))
beta = rng.standard_normal(p)
eps = math.sqrt(s2) * rng.standard_normal((T, n))
y = np.einsum("tnp,p->tn", X, beta) + eps
b = np.linalg.solve(np.einsum("tni,tnj->tij", X, X), np.einsum("tni,tn->ti", X, y)[..., None])[..., 0]
exc = ((b - beta) ** 2).sum(1)
res = y - np.einsum("tnp,tp->tn", X, b)
trn = (res ** 2).mean(1)
report("1.B E excess = s2 p/(n-p-1)", s2 * p / (n - p - 1), exc.mean(), 0.015)
report("1.B E train risk = s2 (n-p)/n", s2 * (n - p) / n, trn.mean(), 0.01)
report("1.B E gap = s2[(n-1)/(n-p-1) - (n-p)/n]", s2 * ((n - 1) / (n - p - 1) - (n - p) / n),
       (s2 + exc - trn).mean(), 0.015)

# ---------- §2.A: NFL brute force ----------
N, m = 6, 3  # domain size, training points (first m inputs)
for learner_name, learner in [("majority", lambda lab: lambda x: int(sum(lab) * 2 >= len(lab))),
                              ("copy-first", lambda lab: lambda x: lab[0]),
                              ("parity-of-x", lambda lab: lambda x: (x + sum(lab)) % 2)]:
    errs = []
    for f in itertools.product([0, 1], repeat=N):
        h = learner(f[:m])
        errs.append(np.mean([h(x) != f[x] for x in range(m, N)]))
    report(f"2.A NFL off-training-set error = 1/2 ({learner_name})", 0.5, float(np.mean(errs)), 1e-12)

# ---------- §2.B: UAT unit count for sin(omega x) ----------
for omega in [10.0, 50.0, 200.0]:
    eps_ = 1e-3
    h = math.sqrt(8 * eps_) / omega
    K = math.ceil(1 / h)
    knots = np.linspace(0, 1, K + 1)
    xs = np.linspace(0, 1, 200_001)
    err = np.abs(np.interp(xs, knots, np.sin(omega * knots)) - np.sin(omega * xs)).max()
    ok = err <= eps_ * (1 + 1e-9)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 2.B piecewise-linear interpolant with K={K} pieces "
          f"(omega={omega}) has sup error {err:.3g} <= eps={eps_}")
# ReLU representation of a piecewise-linear interpolant with K pieces uses K ReLUs + bias
omega, K = 20.0, 40
knots = np.linspace(0, 1, K + 1)
vals = np.sin(omega * knots)
slopes = np.diff(vals) / np.diff(knots)
c = np.concatenate([[slopes[0]], np.diff(slopes)])
xs = np.linspace(0, 1, 10_001)
g = vals[0] + sum(ck * np.maximum(xs - tk, 0) for ck, tk in zip(c, knots[:-1]))
report_abs("2.B ReLU sum reproduces the interpolant (max abs diff)", float(np.abs(g - np.interp(xs, knots, vals)).max()), 1e-10)
# Telgarsky: tent^k has 2^k linear pieces
tent = lambda x: 2 * np.maximum(x, 0) - 4 * np.maximum(x - 0.5, 0)
xs = np.linspace(0, 1, 2 ** 14 + 1)
for k in [1, 2, 3, 5, 8]:
    z = xs.copy()
    for _ in range(k):
        z = tent(z)
    s = np.sign(np.diff(z))
    pieces = 1 + int((s[1:] != s[:-1]).sum())
    report(f"2.B tent^{k} has 2^{k} monotone pieces", 2 ** k, pieces, 0)
# Finite-class bound for W float32 weights, n = 5e4, delta = 0.05
W_, n_, d_ = 1e6, 5e4, 0.05
val = math.sqrt((32 * W_ * math.log(2) + math.log(2 / d_)) / (2 * n_))
print(f"[INFO] 2.C finite-class bound sqrt(log(2|H|/delta)/2n) for W=1e6 float32, n=5e4: {val:.3g}")
report("2.C finite-class bound ~ 14.9", 14.9, val, 0.01)

# ---------- §3: bias-variance, ridge per eigendirection ----------
n, p, s2, lam = 25, 6, 0.5, 0.07
Xf = rng.standard_normal((n, p)) @ np.diag([3, 2, 1, 0.5, 0.2, 0.05])
beta = rng.standard_normal(p)
G = Xf.T @ Xf
s2k, V = np.linalg.eigh(G)
bk = V.T @ beta
T = 200_000
Yf = Xf @ beta + math.sqrt(s2) * rng.standard_normal((T, n))
A = np.linalg.solve(G + n * lam * np.eye(p), Xf.T)
Bh = Yf @ A.T
Ck = (Bh - beta) @ V  # coordinates of error in eigenbasis
bias_mc = Ck.mean(0)
var_mc = Ck.var(0)
bias_cf = -n * lam / (s2k + n * lam) * bk
var_cf = s2 * s2k / (s2k + n * lam) ** 2
for k in range(p):
    report(f"3.B ridge bias_k (k={k})", bias_cf[k], bias_mc[k], 0.01 + 0.003 / max(abs(bias_cf[k]), 1e-3))
    report(f"3.B ridge var_k (k={k})", var_cf[k], var_mc[k], 0.01)
# per-direction optimal n*lam = s2/beta_k^2
sk, bkk = 2.3, 0.8
a = np.logspace(-4, 4, 400_001)
f = (a ** 2 * bkk ** 2 + s2 * sk) / (sk + a) ** 2
report("3.B argmin_a [(a^2 b^2 + s2 s^2)/(s^2+a)^2] = s2/b^2", s2 / bkk ** 2, a[f.argmin()], 1e-3)
# in-sample optimism 2 s2 Tr(S)/n
S = Xf @ A
Yhat = Yf @ S.T
Ynew = Xf @ beta + math.sqrt(s2) * rng.standard_normal((T, n))
opt = (((Ynew - Yhat) ** 2).mean(1) - ((Yf - Yhat) ** 2).mean(1)).mean()
report("3.B optimism = 2 s2 df/n, df = sum s^2/(s^2+n lam)", 2 * s2 * np.sum(s2k / (s2k + n * lam)) / n, opt, 0.02)
report("3.B Tr S = sum s_k^2/(s_k^2 + n lam)", np.sum(s2k / (s2k + n * lam)), np.trace(S), 1e-10)
# Gibbs = Bayes-ensemble + predictive variance (squared loss) -- posterior draws of ridge posterior
tau2 = 1.0
lamB = s2 / (n * tau2)
post_cov = s2 * np.linalg.inv(G + n * lamB * np.eye(p))
mean = np.linalg.solve(G + n * lamB * np.eye(p), Xf.T @ Yf[0])
draws = rng.multivariate_normal(mean, post_cov, size=400_000)
x0 = rng.standard_normal(p)
y0 = x0 @ beta + 0.3
gibbs = ((draws @ x0 - y0) ** 2).mean()
bayes = (mean @ x0 - y0) ** 2
report("3.A Gibbs = Bayes + x0^T Sigma_post x0", bayes + x0 @ post_cov @ x0, gibbs, 0.01)

# ---------- §4: regularization ----------
# MAP posterior covariance
Xf = rng.standard_normal((n, p))
G = Xf.T @ Xf
y1 = Xf @ beta + rng.standard_normal(n)
tau2, sg2 = 0.4, 0.9
lam = sg2 / (n * tau2)
prec = G / sg2 + np.eye(p) / tau2
report_abs("4.A posterior precision = (X^T X + n lam I)/sigma^2", float(np.abs(np.linalg.inv(prec) - sg2 * np.linalg.inv(G + n * lam * np.eye(p))).max()), 1e-12)
report_abs("4.A posterior mean = ridge(lam = s2/(n tau2))", float(np.abs(np.linalg.solve(prec, Xf.T @ y1 / sg2) - np.linalg.solve(G + n * lam * np.eye(p), Xf.T @ y1)).max()), 1e-10)
# MAP non-invariance: Gamma(2,1) mode 1 in beta; mode in u = log beta is beta = 2
u = np.linspace(-3, 3, 600_001)
report("4.A mode of e^{2u - e^u} at beta = e^u = 2", 2.0, float(np.exp(u[np.argmax(2 * u - np.exp(u))])), 1e-4)
# Lagrange: ||beta_lam||^2 strictly decreasing
lams = np.logspace(-4, 2, 200)
norms = [np.sum(np.linalg.solve(G + n * l * np.eye(p), Xf.T @ y1) ** 2) for l in lams]
ok = bool(np.all(np.diff(norms) < 0)); results.append(ok)
print(f"[{'PASS' if ok else 'FAIL'}] 4.B ||beta_lambda||^2 strictly decreasing in lambda")
# Gradient flow filter: integrate GD with tiny step on (1/2n)||X b - y||^2
Xf = rng.standard_normal((n, p)) @ np.diag([2, 1.5, 1, 0.6, 0.3, 0.1])
y1 = Xf @ beta + rng.standard_normal(n)
U, sv, Vt = np.linalg.svd(Xf, full_matrices=False)
t_end, dt = 3.0, 1e-4
bcur = np.zeros(p)
for _ in range(int(t_end / dt)):
    bcur -= dt * Xf.T @ (Xf @ bcur - y1) / n
coef = Vt @ bcur
pred = (1 - np.exp(-sv ** 2 * t_end / n)) * (U.T @ y1) / sv
report_abs("4.C flow filter 1 - exp(-s^2 t/n) (max abs err)", float(np.abs(coef - pred).max()), 1e-4)
lam = 0.37
s2half_ridge = n * lam
s2half_flow = n * math.log(2) / (math.log(2) / lam)
report("4.C half-points match at t = ln2/lambda", s2half_ridge, s2half_flow, 1e-12)
xg = np.logspace(-4, 4, 400_001)
d = (1 - 2 ** (-xg)) - xg / (1 + xg)
report("4.C matched filters: max(flow - ridge) ~ 0.138", 0.138, float(d.max()), 0.01)
report("4.C matched filters: max(ridge - flow) ~ 0.044", 0.0439, float(-d.min()), 0.01)
# GD from 0 on p > n converges to min-norm
nn, pp = 10, 25
Xw = rng.standard_normal((nn, pp)); yw = rng.standard_normal(nn)
bcur = np.zeros(pp)
eta = 0.5 * nn / np.linalg.norm(Xw, 2) ** 2
for _ in range(20000):
    bcur -= eta * Xw.T @ (Xw @ bcur - yw) / nn
report_abs("4.C GD from 0 -> min-norm interpolant (max abs diff)", float(np.abs(bcur - np.linalg.pinv(Xw) @ yw).max()), 1e-8)
# Dropout penalty
q = 0.7
Xd = rng.standard_normal((n, p)) * rng.uniform(0.2, 3, p)
yd = rng.standard_normal(n)
bd = rng.standard_normal(p)
T = 200_000
masks = rng.binomial(1, q, size=(T, n, p)) / q
L = ((yd - np.einsum("np,tnp,p->tn", Xd, masks, bd)) ** 2).mean(1).mean()
closed = ((yd - Xd @ bd) ** 2).mean() + (1 - q) / q * np.sum(bd ** 2 * (Xd ** 2).sum(0)) / n
report("4.D dropout: E loss = loss + (1-q)/q sum_j beta_j^2 ||x_j||^2/n", closed, L, 0.01)
# Flatness: (1/2) Tr(H Sigma) for OLS with x ~ N(0,I): H = 2I -> s2 Tr (X^T X)^{-1}
# (identity check, exact for quadratic risk) -- covered by 1.B
# Dinh rescaling: Tr H = 2a(w1^2 + w2^2) at a minimum of a(w1 w2 - c)^2
a_, c_ = 1.7, 0.9
def L2(w):
    return a_ * (w[0] * w[1] - c_) ** 2
for alpha in [0.1, 1.0, 10.0]:
    w = np.array([alpha * 1.2, c_ / (alpha * 1.2)])
    h = 1e-4
    Hn = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            Hn[i, j] = (L2(w + ei + ej) - L2(w + ei - ej) - L2(w - ei + ej) + L2(w - ei - ej)) / (4 * h * h)
    report(f"4.E Tr H = 2a(w1^2+w2^2) at alpha={alpha}", 2 * a_ * (w ** 2).sum(), float(np.trace(Hn)), 1e-4)
# Invariance of Tr(H Sigma) under linear reparametrization theta = J phi
Hm = rng.standard_normal((p, p)); Hm = Hm @ Hm.T
Sg = rng.standard_normal((p, p)); Sg = Sg @ Sg.T
J = rng.standard_normal((p, p))
Ji = np.linalg.inv(J)
report("4.E Tr(H Sigma) = Tr(J^T H J . J^{-1} Sigma J^{-T})", float(np.trace(Hm @ Sg)),
       float(np.trace(J.T @ Hm @ J @ Ji @ Sg @ Ji.T)), 1e-8)

# ---------- §5: double descent ----------
# finite-n min-norm: bias r^2 (1 - n/p), variance s2 n/(p-n-1), Gaussian isotropic design
r2, s2 = 4.0, 1.0
for (n, p) in [(20, 60), (30, 45)]:
    T = 40_000
    bias = np.empty(T); var = np.empty(T)
    for t in range(T):
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p); beta *= math.sqrt(r2) / np.linalg.norm(beta)
        Xp = np.linalg.pinv(X)
        P = Xp @ X
        bias[t] = np.sum((beta - P @ beta) ** 2)
        var[t] = s2 * np.trace(Xp @ Xp.T)
    report(f"5.B min-norm bias r^2(1-n/p) (n={n},p={p})", r2 * (1 - n / p), bias.mean(), 0.01)
    report(f"5.B min-norm variance s2 n/(p-n-1) (n={n},p={p})", s2 * n / (p - n - 1), var.mean(), 0.03)
# asymptotic branches at large n (single big draws)
def minnorm_risk(n, p, reps):
    out = []
    for _ in range(reps):
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p); beta *= math.sqrt(r2) / np.linalg.norm(beta)
        Xp = np.linalg.pinv(X)
        out.append(np.sum((beta - Xp @ X @ beta) ** 2) + s2 * np.trace(Xp @ Xp.T))
    return float(np.mean(out))
for g in [0.3, 0.6, 2.0, 4.0]:
    n = 400; p = int(g * n)
    closed = s2 * g / (1 - g) if g < 1 else r2 * (1 - 1 / g) + s2 / (g - 1)
    report(f"5.B asymptotic min-norm risk at gamma={g} (n=400)", closed, minnorm_risk(n, p, 6), 0.05)
# second-descent minimum
r, sg = math.sqrt(r2), math.sqrt(s2)
gs = np.linspace(1.0001, 50, 2_000_001)
Rg = r2 * (1 - 1 / gs) + s2 / (gs - 1)
report("5.C argmin gamma = r/(r - sigma)", r / (r - sg), float(gs[Rg.argmin()]), 1e-4)
report("5.C min risk = 2 r sigma - sigma^2 (< r^2)", 2 * r * sg - s2, float(Rg.min()), 1e-6)
# optimal ridge lambda* = s2 gamma / r^2 is Bayes for beta ~ N(0, r^2/p I); risk nonincreasing in n at fixed p
p = 60
lst = []
for n in [20, 40, 55, 60, 65, 80, 120, 240]:
    T = 3000; acc = 0.0
    lamst = s2 * (p / n) / r2
    for _ in range(T):
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p) * math.sqrt(r2 / p)
        y = X @ beta + math.sqrt(s2) * rng.standard_normal(n)
        b = np.linalg.solve(X.T @ X + n * lamst * np.eye(p), X.T @ y)
        acc += np.sum((b - beta) ** 2)
    lst.append(acc / T)
ok = all(lst[i + 1] < lst[i] for i in range(len(lst) - 1)); results.append(ok)
print(f"[{'PASS' if ok else 'FAIL'}] 5.C optimal-ridge risk decreasing in n at p=60: {np.round(lst, 3)}")
# min-norm at same (n,p) for contrast (finite-n peak)
print("[INFO] 5.C min-norm risk at p=60 for n in [20,40,55,65,80,120,240]:",
      [round(minnorm_risk(n, p, 300), 2) for n in [20, 40, 55, 65, 80, 120, 240]])

print(f"\n{sum(results)}/{len(results)} checks passed")
