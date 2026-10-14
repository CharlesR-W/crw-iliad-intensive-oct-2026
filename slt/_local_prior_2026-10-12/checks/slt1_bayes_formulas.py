#!/usr/bin/env python3
"""Checks for every formula stated in slt1_bayes (NB1). Run from slt/: python checks/slt1_bayes_formulas.py

Each block prints what it checks and asserts it. Symbolic checks use sympy; the rest are
numerical (quadrature, exact sums, or Monte Carlo with stated tolerances).
"""
import numpy as np
import sympy as sp
from scipy.special import gammaln, expit, log_expit, logsumexp
from scipy.stats import norm, expon, gamma as gamma_dist, lognorm
from scipy.integrate import quad

rng = np.random.default_rng(12345)
ok = lambda msg: print("OK  ", msg)

# ---------------------------------------------------------------- §1.1 thermo identities
# 1-D model: p(x|w) = N(x; tanh(w)*2, 1) (non-Gaussian posterior), prior N(0,1), small sample.
X = rng.normal(0.7, 1.0, size=15)
n = len(X)
wg = np.linspace(-8, 8, 200001)
logphi = norm.logpdf(wg)
nLn = -norm.logpdf(X[:, None], 2 * np.tanh(wg)[None, :]).sum(0)

def F(beta):
    lw = logphi - beta * nLn
    m = lw.max()
    return -(m + np.log(np.trapezoid(np.exp(lw - m), wg)))

def post(beta):
    lw = logphi - beta * nLn
    p = np.exp(lw - lw.max()); return p / np.trapezoid(p, wg)

for b in [0.3, 1.0, 2.5]:
    h = 1e-4
    dF = (F(b + h) - F(b - h)) / (2 * h)
    d2F = (F(b + h) - 2 * F(b) + F(b - h)) / h ** 2
    p = post(b)
    E = np.trapezoid(p * nLn, wg); V = np.trapezoid(p * nLn ** 2, wg) - E ** 2
    assert abs(dF - E) < 1e-5 * max(1, abs(E)), (dF, E)
    assert abs(d2F + V) < 1e-3 * max(1, V), (d2F, V)
ok("F'(beta) = E_beta[nL_n], F''(beta) = -Var_beta[nL_n]")
assert abs(F(0.0)) < 1e-9
bs = np.linspace(0, 1, 2001)
Es = [np.trapezoid(post(b) * nLn, wg) for b in bs]
assert abs(np.trapezoid(Es, bs) - F(1.0)) < 1e-5
ok("thermodynamic integration F(1) = int_0^1 E_beta[nL_n] dbeta")
for b in [0.5, 1.0]:
    for m_, s_ in [(0.3, 0.5), (1.0, 0.2), (-1, 1.5)]:
        Zr = np.trapezoid(norm.pdf(wg, m_, s_), wg)                       # renormalise on the grid
        rho, logrho = norm.pdf(wg, m_, s_) / Zr, norm.logpdf(wg, m_, s_) - np.log(Zr)
        lhs = b * np.trapezoid(rho * nLn, wg) + np.trapezoid(rho * (logrho - logphi), wg)
        logp = logphi - b * nLn + F(b)                                    # log p_beta, exactly
        kl = np.trapezoid(rho * (logrho - logp), wg)
        assert abs(lhs - (F(b) + kl)) < 1e-6, (lhs, F(b) + kl)
        assert lhs >= F(b)
ok("beta E_rho[nL_n] + KL(rho||phi) = F + KL(rho||p_beta) >= F (Gaussian rho)")

# Ex 1.1a: F = -log sum exp(-F_j), F_1 - log k <= F <= F_1
cuts = [-9, -1, 0.2, 1.5, 9]
dw = wg[1] - wg[0]
def Friemann(sel):
    lw = logphi[sel] - nLn[sel]; m = lw.max()
    return -(m + np.log(np.exp(lw - m).sum() * dw))      # Riemann sums: exactly additive over a partition
Fj = np.array([Friemann((wg >= lo) & (wg < hi)) for lo, hi in zip(cuts[:-1], cuts[1:])])
Ftot = Friemann(np.ones_like(wg, bool))
assert abs(-logsumexp(-Fj) - Ftot) < 1e-10 and abs(Ftot - F(1.0)) < 1e-6
assert Fj.min() - np.log(len(Fj)) <= Ftot <= Fj.min()
ok(f"Ex 1.1a decomposition and bounds (F = {F(1.0):.4f}, local F = {np.round(Fj, 3)})")
assert abs(np.exp(5) - 148.4) < 0.1
ok("e^5 ~ 150")

# ---------------------------------------------------------------- §1.2
# Ex 1.2a coverage numbers
cov2 = 2 * norm.cdf(1.96 / 2) - 1
covh = 2 * norm.cdf(1.96 / 0.5) - 1
assert abs(cov2 - 0.67) < 0.005 and covh > 0.9999
ok(f"coverage sigma=2: {cov2:.4f}; sigma=1/2: {covh:.6f}")
# Monte Carlo: posterior N(xbar, 1/n) exactly; coverage
sig, nn, R = 2.0, 50, 200000
xbar = rng.normal(0, sig / np.sqrt(nn), R)
assert abs(np.mean(np.abs(xbar) <= 1.96 / np.sqrt(nn)) - cov2) < 0.005
ok("coverage confirmed by simulation")
# information identity C = H for a realisable model (Poisson in log-rate, Bernoulli logit): symbolic
w, x = sp.symbols("w x", real=True)
lam = sp.exp(w)
score = x - lam                         # d/dw log Pois(x; e^w)
hess = -sp.diff(score, w)               # -d2/dw2 log p = e^w
# under q = Pois(e^w): E x = e^w, Var x = e^w
EH = hess
C = lam                                 # Var of score = Var x
assert sp.simplify(EH - C) == 0
ok("information identity C = H for Poisson (symbolic)")

# ---------------------------------------------------------------- §1.3 Laplace / BIC / volume
d, t, nn_ = sp.symbols("d t n", positive=True)
omega = sp.pi ** (d / 2) / sp.gamma(d / 2 + 1)
assert sp.simplify(omega * sp.gamma(d / 2 + 1) - sp.pi ** (d / 2)) == 0
for dv in [1, 2, 3, 5]:
    V = omega.subs(d, dv) * (2 * t) ** sp.Rational(dv, 2)       # phi(w0)=1, det H = 1
    Zbar = nn_ * sp.integrate(sp.exp(-nn_ * t) * V, (t, 0, sp.oo))
    assert sp.simplify(Zbar - (2 * sp.pi / nn_) ** sp.Rational(dv, 2)) == 0
    assert sp.simplify(Zbar - sp.gamma(sp.Rational(dv, 2) + 1) * V.subs(t, 1 / nn_)) == 0
ok("Zbar_n = n int e^{-nt} V(t) dt = (2pi/n)^{d/2} = Gamma(d/2+1) V(1/n) for d = 1,2,3,5")

# Laplace on a non-Gaussian 2-D integral: error O(1/n)
def lap_err(nv):
    # integrand phi(w) exp(-n f(w)), f(w) = cosh(w1)-1 + (w2 - 0.3 w1^2)^2/2 + 0.1 w2^4 ; phi = N(0, I)
    g = np.linspace(-6, 6, 1201); A, B = np.meshgrid(g, g, indexing="ij")
    f = np.cosh(A) - 1 + (B - 0.3 * A ** 2) ** 2 / 2 + 0.1 * B ** 4
    lw = -nv * f + norm.logpdf(A) + norm.logpdf(B)
    m = lw.max(); Fex = -(m + np.log(np.trapezoid(np.trapezoid(np.exp(lw - m), g, axis=1), g)))
    # minimum at 0, Hessian diag(1, 1) (d2/dw1^2 cosh = 1; cross terms vanish at 0)
    Flap = 0 + np.log(nv / (2 * np.pi)) + 0.5 * np.log(1.0) - 2 * norm.logpdf(0)
    return Fex - Flap
e1, e2 = lap_err(50), lap_err(200)
assert abs(e1 / e2 - 4) < 0.6, (e1, e2)
ok(f"Laplace error ~ 1/n (n=50: {e1:.4f}, n=200: {e2:.4f}, ratio {e1 / e2:.2f})")

# Ex 1.3b coin: Stirling version of -log Z equals Laplace exactly (with log(n+1) -> log n)
k_, n_ = sp.symbols("k n", positive=True)
st = lambda m: m * sp.log(m) - m + sp.log(2 * sp.pi * m) / 2
exact_st = st(n_) + sp.log(n_) - st(k_) - st(n_ - k_)
th = k_ / n_
laplace = -k_ * sp.log(th) - (n_ - k_) * sp.log(1 - th) + sp.log(n_ / (2 * sp.pi * th * (1 - th))) / 2
diff = sp.simplify(sp.expand_log(exact_st - laplace, force=True))
assert diff == 0, diff
ok("coin: Stirling(-log Z) == Laplace (symbolic)")
for nv, kv in [(10, 3), (100, 30), (1000, 300)]:
    ex = -(gammaln(kv + 1) + gammaln(nv - kv + 1) - gammaln(nv + 2))
    t_ = kv / nv
    lp = -(kv * np.log(t_) + (nv - kv) * np.log(1 - t_)) + 0.5 * np.log(nv / (2 * np.pi * t_ * (1 - t_)))
    print(f"     coin n={nv}: exact - Laplace = {ex - lp:.2e}")
# BIC offsets quoted on the page
c_coin = 0.5 * np.log(1 / (0.3 * 0.7)) - 0.5 * np.log(2 * np.pi)
print(f"     coin BIC offset (exact - BIC -> 1/2 log det H - log phi - d/2 log 2pi) = {c_coin:.3f}")
# logistic at w_true=(0.5,1.5), x~N(0,1), prior N(0,4I): H = E[s(1-s) (1,x)(1,x)^T]
xs = np.linspace(-10, 10, 200001); px = norm.pdf(xs); s = expit(0.5 + 1.5 * xs); v = s * (1 - s)
H = np.array([[np.trapezoid(px * v, xs), np.trapezoid(px * v * xs, xs)],
              [np.trapezoid(px * v * xs, xs), np.trapezoid(px * v * xs ** 2, xs)]])
logphi0 = -0.5 * (0.25 + 2.25) / 4 - np.log(2 * np.pi * 4)
c_log = 0.5 * np.log(np.linalg.det(H)) - logphi0 - np.log(2 * np.pi)
print(f"     logistic BIC offset = {c_log:.3f}")
assert -0.16 < c_coin < -0.12 and -0.45 < c_log < -0.3
ok("BIC offsets: coin ~ -0.14, logistic ~ -0.4 (exact below BIC)")

# ---------------------------------------------------------------- §1.4
# Ex 1.4a(1): E G_n = E F_{n+1} - E F_n - S, coin, exact sums over the binomial
theta0 = 0.3
S = -(theta0 * np.log(theta0) + (1 - theta0) * np.log(1 - theta0))
def EF(nv):
    k = np.arange(nv + 1)
    logpk = gammaln(nv + 1) - gammaln(k + 1) - gammaln(nv - k + 1) + k * np.log(theta0) + (nv - k) * np.log(1 - theta0)
    Fk = -(gammaln(k + 1) + gammaln(nv - k + 1) - gammaln(nv + 2))
    return np.sum(np.exp(logpk) * Fk)
def EG(nv):
    k = np.arange(nv + 1)
    logpk = gammaln(nv + 1) - gammaln(k + 1) - gammaln(nv - k + 1) + k * np.log(theta0) + (nv - k) * np.log(1 - theta0)
    ph = (k + 1) / (nv + 2)                                     # Laplace's rule of succession
    kl = theta0 * np.log(theta0 / ph) + (1 - theta0) * np.log((1 - theta0) / (1 - ph))
    return np.sum(np.exp(logpk) * kl)
for nv in [0, 5, 50, 500]:
    assert abs(EG(nv) - (EF(nv + 1) - EF(nv) - S)) < 1e-10
ok("E G_n = E F_{n+1} - E F_n - S (coin, exact)")
for nv in [200, 2000]:
    print(f"     coin n={nv}: n E G_n = {nv * EG(nv):.4f} (d/2 = 0.5)")
assert abs(2000 * EG(2000) - 0.5) < 0.01
ok("n E G_n -> d/2 for the coin")
assert abs(sum(EG(m) for m in range(300)) - (EF(300) - 300 * S)) < 1e-9
ok("E F_n - nS = sum_{m<n} E G_m")

# Ex 1.4a(3): E[n L_n(w_hat)] = n L(w0) - tr(H^-1 C)/2 ; exponential model on lognormal data
# model p(x|w) = e^w exp(-e^w x): l = -w + e^w x; w_hat = -log xbar; nL_n(w_hat) = n(log xbar + 1)
sigl = 0.8; q = lognorm(s=sigl)
m1, var = q.mean(), q.var()
w0 = -np.log(m1); L0 = -w0 + 1
trHC = var / m1 ** 2                   # H = e^{w0} m1 = 1, C = e^{2 w0} Var x
for nv in [100]:
    Xs = q.rvs(size=(100000, nv), random_state=rng); xb = Xs.mean(1)
    val = (nv * (np.log(xb) + 1)).mean() - nv * L0
    se = (nv * (np.log(xb) + 1)).std() / np.sqrt(len(xb))
    print(f"     E[nL_n(w_hat)] - nL(w0) = {val:.3f} +- {se:.3f}, -tr/2 = {-trHC / 2:.3f}")
    assert abs(val + trHC / 2) < 4 * se + 0.03
ok("E[nL_n(w_hat)] = nL(w0) - tr(H^-1 C)/2 + o(1)")

# Ex 1.4b: the table at general beta, misspecified exponential model, exact posterior on a grid
def errors(nv, beta, R=20000):
    wgr = np.linspace(w0 - 1.2, w0 + 1.2, 801)
    lphi = norm.logpdf(wgr, w0 + 0.2, 1.0)
    Xs = q.rvs(size=(R, nv), random_state=rng); sx = Xs.sum(1)
    lp = lphi[None, :] + beta * (nv * wgr[None, :] - np.exp(wgr)[None, :] * sx[:, None])
    P = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    Lw = -wgr + np.exp(wgr) * m1                                   # L(w)
    gibbs = P @ Lw - L0
    mle = (np.log(sx / nv) + m1 * nv / sx) - L0                # L(w_hat), w_hat = -log xbar
    # Bayes predictive excess: E_q[-log sum_j P_j e^{w_j} e^{-e^{w_j} x}] - L0, by quadrature over x
    xq = np.linspace(1e-6, 40, 20000); qx = q.pdf(xq)
    logp0 = w0 - np.exp(w0) * xq                                    # log p(x|w0): subtract inside the integral
    bay = np.empty(R)
    ew = np.exp(wgr)
    for i0 in range(0, R, 500):
        pred = (P[i0:i0 + 500] * ew[None, :]) @ np.exp(-np.outer(ew, xq))     # (r, xq)
        bay[i0:i0 + 500] = np.trapezoid(qx[None, :] * (logp0[None, :] - np.log(pred)), xq, axis=1)
    return nv * mle.mean(), nv * bay.mean(), nv * gibbs.mean(), nv * bay.std() / np.sqrt(R)
lam, nu = 0.5, trHC / 2
# finite-n check: a few percent agreement at n = 400
for beta in [0.5, 1.0, 3.0]:
    mle, bay, gib, se = errors(400, beta)
    tb, tg = (lam - nu) / beta + nu, lam / beta + nu
    print(f"     beta={beta}: n*MLE {mle:.3f} [{nu:.3f}]  n*Bayes {bay:.3f}+-{se:.3f} [{tb:.3f}]  n*Gibbs {gib:.3f} [{tg:.3f}]")
    assert abs(mle - nu) < 0.06 and abs(bay - tb) < 4 * se + 0.04 and abs(gib - tg) < 0.06
ok("regular-case table (MLE nu, Bayes (lam-nu)/beta+nu, Gibbs lam/beta+nu), misspecified exponential model")

# Ex 1.4c exact formulas (Gaussian location, wrong variance), symbolic in sigma, n
sg, nn2 = sp.symbols("sigma n", positive=True)
Exb2 = sg ** 2 / nn2
loss = lambda m2, s2: sp.log(2 * sp.pi * s2) / 2 + (sg ** 2 + m2) / (2 * s2)
L0g = loss(0, 1)
assert sp.simplify(loss(Exb2, 1) - L0g - sg ** 2 / (2 * nn2)) == 0                          # MLE
assert sp.simplify(loss(Exb2, 1 + 1 / nn2) - L0g - sp.log(1 + 1 / nn2) / 2) == 0            # Bayes
assert sp.simplify((Exb2 + 1 / nn2) / 2 - (1 + sg ** 2) / (2 * nn2)) == 0                    # Gibbs
Sg = sp.log(2 * sp.pi * sp.E * sg ** 2) / 2
assert sp.simplify(L0g - Sg - (sg ** 2 - 1 - sp.log(sg ** 2)) / 2) == 0
assert sp.series(sp.log(1 + 1 / nn2) / 2, nn2, sp.oo, 3).removeO().equals(1 / (2 * nn2) - 1 / (4 * nn2 ** 2))
ok("Ex 1.4c exact excesses: sigma^2/2n, log(1+1/n)/2, (1+sigma^2)/2n; gap (sigma^2-1-log sigma^2)/2")
# and by simulation for one sigma
sg_, n3, R = 0.6, 20, 400000
xb = rng.normal(0, sg_ / np.sqrt(n3), R)
bay = (np.log(2 * np.pi * (1 + 1 / n3)) / 2 + (sg_ ** 2 + xb ** 2) / (2 * (1 + 1 / n3))).mean() - (np.log(2 * np.pi) / 2 + sg_ ** 2 / 2)
assert abs(bay - np.log(1 + 1 / n3) / 2) < 5e-4
ok("Ex 1.4c Bayes excess confirmed by simulation")

# Ex 1.4c(3): exact threshold sigma^2 < n log(1+1/n), log 2 at n = 1, increasing to 1
thr = [nv * np.log1p(1 / nv) for nv in [1, 2, 10, 100, 10000]]
assert abs(thr[0] - np.log(2)) < 1e-12 and all(np.diff(thr) > 0) and abs(thr[-1] - 1) < 1e-4
ok("Ex 1.4c exact MLE-vs-Bayes threshold n log(1+1/n): log 2 at n=1, increasing to 1")

# §1.2 code: TV between Beta(k+1, n-k+1) and N(k/n, t(1-t)/n), including Gaussian mass outside [0,1]
from scipy.stats import beta as beta_dist
def tv(nv, kv):
    t = kv / nv; sd = np.sqrt(t * (1 - t) / nv); th = np.linspace(0, 1, 400001)
    inner = 0.5 * np.trapezoid(np.abs(beta_dist.pdf(th, kv + 1, nv - kv + 1) - norm.pdf(th, t, sd)), th)
    return inner + 0.5 * (norm.cdf(0, t, sd) + norm.sf(1, t, sd))
v = tv(10, 2); assert abs(v - 0.1269) < 1e-3
ok(f"TV(n=10,k=2) = {v:.4f} (Codex's independent value 0.126925)")
print("all checks passed")
