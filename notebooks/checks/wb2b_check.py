#!/usr/bin/env python3
"""Numerical checks for WB2 part B (sections 6-9 of 02_statistical_learning).

Run: python checks/wb2b_check.py   (numpy + scipy; about a minute)

Each block prints the formula used in the notebook next to an independent
numerical value. Conventions match the notebook: loss l, population risk R,
H = Hess R(theta*), C = Cov[grad l(theta*)], squared loss is (1/2)(y - f)^2,
log loss is in nats.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import norm

rng = np.random.default_rng(0)
OK = True


def report(name, formula, numeric, rtol=0.05, atol=0.0):
    global OK
    good = np.allclose(formula, numeric, rtol=rtol, atol=atol)
    OK &= bool(good)
    f, v = np.asarray(formula, float), np.asarray(numeric, float)
    if f.size > 4:
        err = np.max(np.abs(f - v)) / np.max(np.abs(f))
        print(f"  [{'ok' if good else 'FAIL'}] {name}: max |formula - numeric| / max|formula| = {err:.1e}")
    else:
        print(f"  [{'ok' if good else 'FAIL'}] {name}: formula={np.round(f, 5)}  numeric={np.round(v, 5)}")


# ---------------------------------------------------------------------------
print("\n§7.1-7.2  Sandwich, excess risk and TIC optimism: misspecified regression")
# x ~ N(0,1), features phi(x) = (1, x), y = sin(2x) + (0.3 + 0.5|x|) * noise.
# Linear model is misspecified (wrong mean) and the noise is heteroscedastic.
def sample(n, rng):
    x = rng.standard_normal(n)
    y = np.sin(2 * x) + (0.3 + 0.5 * np.abs(x)) * rng.standard_normal(n)
    return np.stack([np.ones(n), x], 1), y

Phi, Y = sample(4_000_000, rng)
H = Phi.T @ Phi / len(Y)                       # Hessian of (1/2)E(y - phi.theta)^2
theta_star = np.linalg.solve(H, Phi.T @ Y / len(Y))
r = Phi @ theta_star - Y
G = r[:, None] * Phi                           # per-sample gradients at theta*
C = G.T @ G / len(Y)
Hinv = np.linalg.inv(H)
sandwich = Hinv @ C @ Hinv
tr = np.trace(Hinv @ C)
sigma2_naive = np.mean(r ** 2)                 # "sigma^2 p" would be the well-specified answer
print(f"  Tr(H^-1 C) = {tr:.4f}   vs naive sigma^2 p = {sigma2_naive * 2:.4f}")

n, reps = 200, 20000
est, exc, opt = [], [], []
for _ in range(reps):
    P, y = sample(n, rng)
    th = np.linalg.lstsq(P, y, rcond=None)[0]
    d = th - theta_star
    est.append(np.sqrt(n) * d)
    exc.append(0.5 * d @ H @ d)                # R(th) - R(theta*), exact for a quadratic loss
    train = 0.5 * np.mean((y - P @ th) ** 2)
    R_th = 0.5 * np.mean(r ** 2) + 0.5 * d @ H @ d
    opt.append(R_th - train)
est = np.array(est)
report("Cov sqrt(n)(th - th*) = H^-1 C H^-1", sandwich.ravel(), np.cov(est.T).ravel(), rtol=0.05, atol=0.01)
report("E excess risk = Tr(H^-1 C)/(2n)", tr / (2 * n), np.mean(exc), rtol=0.05)
report("E optimism   = Tr(H^-1 C)/n", tr / n, np.mean(opt), rtol=0.05)
print(f"  ratio optimism/excess = {np.mean(opt) / np.mean(exc):.3f}  (theory 2)")

print("\n§7.2  Same checks with a non-quadratic loss: misspecified logistic regression")
# x ~ N(0,1), P(y=1|x) = sigmoid(3 x^3) (not in the linear-logit family), model theta.(1,x).
# Population quantities by quadrature over x (exact in y), so only the n-sample fits are random.
def lsample(n, rng):
    x = rng.standard_normal(n)
    y = (rng.random(n) < expit(3 * x ** 3)).astype(float)
    return np.stack([np.ones(n), x], 1), y

xq = np.linspace(-9, 9, 6001)
wq = norm.pdf(xq) * (xq[1] - xq[0])
Pq = np.stack([np.ones_like(xq), xq], 1)
pq = expit(3 * xq ** 3)

def R_pop(th):
    z = Pq @ th
    return np.sum(wq * (np.logaddexp(0, z) - pq * z))

def fit_logistic(P, y, w=None, th=None):
    w = np.full(len(y), 1.0 / len(y)) if w is None else w
    th = np.zeros(P.shape[1]) if th is None else th.copy()
    for _ in range(100):
        s = expit(P @ th)
        g = P.T @ (w * (s - y))
        Hh = (P * (w * s * (1 - s))[:, None]).T @ P
        step = np.linalg.solve(Hh, g)
        th -= step
        if np.max(np.abs(step)) < 1e-13:
            break
    return th

def logloss(P, y, th):
    z = P @ th
    return np.mean(np.logaddexp(0, z) - y * z)

thL = fit_logistic(Pq, pq, wq)                  # population minimizer
s = expit(Pq @ thL)
HL = (Pq * (wq * s * (1 - s))[:, None]).T @ Pq
CL = (Pq * (wq * (s ** 2 - 2 * s * pq + pq))[:, None]).T @ Pq   # E_y (s - y)^2 = s^2 - 2 s p + p
trL = np.trace(np.linalg.solve(HL, CL))
RL_star = R_pop(thL)
print(f"  Tr(H^-1 C) = {trL:.4f}  (would be p = 2 if well specified)")
n, reps = 400, 20000
excL, optL = [], []
for _ in range(reps):
    P, y = lsample(n, rng)
    th = fit_logistic(P, y, th=thL)
    e = R_pop(th) - RL_star
    excL.append(e)
    # optimism = excess + [Rhat(th*) - Rhat(th)] + [R(th*) - Rhat(th*)]; the last bracket has
    # mean exactly zero, so it is dropped to reduce Monte Carlo variance.
    optL.append(e + logloss(P, y, thL) - logloss(P, y, th))
report("E excess risk = Tr(H^-1 C)/(2n)", trL / (2 * n), np.mean(excL), rtol=0.06)
report("E optimism   = Tr(H^-1 C)/n", trL / n, np.mean(optL), rtol=0.06)
print(f"  ratio optimism/excess = {np.mean(optL) / np.mean(excL):.3f}  (theory 2)")

print("\n§7.3  Fisher identity C = H for a well-specified logistic model (AIC)")
x = rng.standard_normal(2_000_000)
P = np.stack([np.ones_like(x), x], 1)
th0 = np.array([0.3, 1.5])
y = (rng.random(len(x)) < expit(P @ th0)).astype(float)
s = expit(P @ th0)
Hw = (P * (s * (1 - s))[:, None]).T @ P / len(x)
Gw = (s - y)[:, None] * P
Cw = Gw.T @ Gw / len(x)
report("C = H (entries)", Hw.ravel(), Cw.ravel(), rtol=0.02, atol=1e-3)

print("\n§7.5  Bayes vs Gibbs generalization error, regular well-specified case")
# y = x.beta + eps, eps ~ N(0, s2) with s2 known, flat prior: posterior N(bhat, s2 (X'X)^-1).
# Generalization errors in nats per sample, measured as KL from truth, averaged over x ~ N(0, I).
p, n, s2, reps = 5, 100, 1.0, 4000
beta = rng.standard_normal(p)
bayes, gibbs, mle = [], [], []
for _ in range(reps):
    X = rng.standard_normal((n, p))
    yv = X @ beta + np.sqrt(s2) * rng.standard_normal(n)
    XtX = X.T @ X
    bh = np.linalg.solve(XtX, X.T @ yv)
    Sp = s2 * np.linalg.inv(XtX)
    d = bh - beta
    # KL(N(m0,s2) || N(m1, v1)) averaged over x ~ N(0,I):
    # predictive at x: N(x.bh, s2 + x'Sp x). E_x of KL computed by MC over x.
    xs = rng.standard_normal((400, p))
    v1 = s2 + np.einsum("ij,jk,ik->i", xs, Sp, xs)
    m = xs @ d
    kl = 0.5 * (np.log(v1 / s2) + (s2 + m ** 2) / v1 - 1)
    bayes.append(kl.mean())
    mle.append(0.5 * d @ d / s2)              # E_x KL(N(x.beta,s2)||N(x.bh,s2)) = |d|^2/(2 s2)
    u = np.linalg.cholesky(Sp) @ rng.standard_normal(p)
    gibbs.append(0.5 * (d + u) @ (d + u) / s2)
# finite-n exact values: E|d|^2 = s2 * p/(n-p-1); include for reference
print(f"  (exact finite-n MLE value p/(2(n-p-1)) = {p / (2 * (n - p - 1)):.5f})")
report("MLE excess  ~ p/(2n)", p / (2 * n), np.mean(mle), rtol=0.1)
report("Bayes error ~ p/(2n)", p / (2 * n), np.mean(bayes), rtol=0.1)
report("Gibbs error ~ p/n", p / n, np.mean(gibbs), rtol=0.1)

# ---------------------------------------------------------------------------
print("\n§8.1  Diagonal envelope  H_ii^2 <= C_ii <= G^2 H_ii  for C = E[|v|^2 v v^T], H = E[v v^T]")
V = rng.standard_normal((5000, 30)) * rng.lognormal(0, 1, 30)
Hd = np.mean(V ** 2, 0)
Cd = np.mean((V ** 2).sum(1)[:, None] * V ** 2, 0)
G2 = (V ** 2).sum(1).max()
ok_env = np.all(Hd ** 2 <= Cd + 1e-12) and np.all(Cd <= G2 * Hd + 1e-12)
OK &= bool(ok_env)
print(f"  [{'ok' if ok_env else 'FAIL'}] envelope holds on random data")
# frequency construction (gamma=1) vs magnitude construction (gamma=2)
k = np.arange(1, 21)
pi_i, s_i = 1.0 / k, np.ones_like(k, float)          # usage frequency varies
g1 = np.polyfit(np.log(pi_i * s_i ** 2), np.log(pi_i * s_i ** 4), 1)[0]
pi_i, s_i = np.ones_like(k, float) / 20, 1.0 / k      # gradient magnitude varies
g2 = np.polyfit(np.log(pi_i * s_i ** 2), np.log(pi_i * s_i ** 4), 1)[0]
report("one-coordinate-per-sample: frequency varies -> gamma", 1.0, g1, rtol=1e-6)
report("one-coordinate-per-sample: magnitude varies -> gamma", 2.0, g2, rtol=1e-6)

print("\n§8.2  Lyapunov equation H S + S H = (eta/B) C")
eta, B, pdim = 0.1, 32, 6
Q = np.linalg.qr(rng.standard_normal((pdim, pdim)))[0]
h = np.exp(rng.uniform(-3, 2, pdim))
Hm = Q @ np.diag(h) @ Q.T
for gamma in [0.0, 1.0, 1.5, 2.0]:
    Cm = Q @ np.diag(h ** gamma) @ Q.T
    S = solve_continuous_lyapunov(Hm, (eta / B) * Cm)
    closed = (eta / (2 * B)) * Cm @ np.linalg.inv(Hm)
    report(f"commuting closed form, gamma={gamma}", closed.ravel(), S.ravel(), rtol=1e-8, atol=1e-12)
# Non-commuting C
A = rng.standard_normal((pdim, pdim))
Cnc = A @ A.T
S = solve_continuous_lyapunov(Hm, (eta / B) * Cnc)
report("1/2 Tr(H S) = eta Tr C / (4B), non-commuting C", eta * np.trace(Cnc) / (4 * B), 0.5 * np.trace(Hm @ S), rtol=1e-10)
# C proportional to I => S proportional to H^-1, even with random H
S = solve_continuous_lyapunov(Hm, (eta / B) * 2.7 * np.eye(pdim))
report("C = cI  =>  S = (eta c / 2B) H^-1", (eta * 2.7 / (2 * B) * np.linalg.inv(Hm)).ravel(), S.ravel(), rtol=1e-8, atol=1e-12)
# Non-commuting C not ∝ I => H S is not ∝ I (no temperature matches)
S = solve_continuous_lyapunov(Hm, (eta / B) * Cnc)
ev = np.linalg.eigvals(Hm @ S).real
print(f"  non-commuting C: eigenvalue spread of H S = {ev.max() / ev.min():.2f} (1 would mean posterior shape)")

# Discrete SGD simulation in 2-D as an independent check of the SDE law at small eta
H2 = np.diag([1.0, 5.0])
C2 = np.diag([1.0, 5.0 ** 1.5])
eta2, B2 = 0.01, 1
S_sde = solve_continuous_lyapunov(H2, (eta2 / B2) * C2)
th = np.zeros((20000, 2))
L = np.linalg.cholesky(C2 / B2)
for t in range(3000):
    th = th - eta2 * (th @ H2 + rng.standard_normal(th.shape) @ L.T)
report("simulated SGD covariance vs Lyapunov (O(eta) differences expected)", S_sde.ravel(), np.cov(th.T).ravel(), rtol=0.08, atol=2e-4)

print("\n§8.3  KL-optimal temperature and the gamma symmetry")
def kl_gauss(S1, S2):
    p = len(S1)
    S2i = np.linalg.inv(S2)
    return 0.5 * (np.trace(S2i @ S1) - p + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])

nN = 1000
Cm = Q @ np.diag(h ** 1.5) @ Q.T
S = solve_continuous_lyapunov(Hm, (eta / B) * Cm)
Tstar = nN * np.trace(Hm @ S) / pdim
res = minimize_scalar(lambda lt: kl_gauss(S, np.exp(lt) / nN * np.linalg.inv(Hm)), bounds=(-10, 10), method="bounded")
report("T* = n Tr(H S)/p", Tstar, np.exp(res.x), rtol=1e-4)
klstar = 0.5 * (pdim * np.log(np.trace(Hm @ S) / pdim) - np.linalg.slogdet(Hm @ S)[1])
report("KL* = 1/2[p log(Tr(HS)/p) - log det(HS)]", klstar, res.fun, rtol=1e-6)
report("posterior excess T* p/(2n) = SGD excess 1/2 Tr(HS)", Tstar * pdim / (2 * nN), 0.5 * np.trace(Hm @ S), rtol=1e-10)

def klstar_gamma(hs, g):
    e = hs ** g
    return 0.5 * (len(hs) * np.log(e.mean()) - np.log(e).sum())

h2 = np.array([1.0, 7.0])
report("p=2: KL*(gamma) = KL*(-gamma)", klstar_gamma(h2, 1.5), klstar_gamma(h2, -1.5), rtol=1e-10)
h3 = np.array([1.0, 2.0, 30.0])
a, b = klstar_gamma(h3, 1.5), klstar_gamma(h3, -1.5)
print(f"  p=3, spectrum (1,2,30): KL*(1.5) = {a:.4f}, KL*(-1.5) = {b:.4f}  -> symmetry is special to p=2 / symmetric log-spectra")

# ---------------------------------------------------------------------------
print("\n§6(b)  Binned entropy of tanh(aZ) and ReLU(aZ), Z ~ N(0,1), exact bin probabilities")
K = 30
edges = np.linspace(-1, 1, K + 1)

def H_tanh(a):
    zc = np.arctanh(np.clip(edges[1:-1], -1 + 1e-15, 1 - 1e-15)) / a
    cdf = np.concatenate([[0.0], norm.cdf(zc), [1.0]])
    pr = np.diff(cdf)
    pr = pr[pr > 0]
    return -(pr * np.log(pr)).sum()

def H_relu(a, delta=2.0 / K, nbins=200000):
    e = np.arange(nbins + 1) * delta / a          # bins of width delta in T = a z, z>0
    cdf = norm.cdf(e)
    pr = np.diff(cdf)
    pr[0] += 0.5                                  # the atom at T = 0 (all z <= 0)
    pr = pr[pr > 0]
    return -(pr * np.log(pr)).sum()

for a in [0.1, 0.5, 1, 2, 5, 20, 100, 1e4]:
    print(f"  a={a:>8}: H_bin(tanh) = {H_tanh(a):.4f}   H_bin(ReLU) = {H_relu(a):.4f}")
report("tanh: H -> log 2 as a -> inf", np.log(2), H_tanh(1e6), rtol=1e-4)
hplus = 0.5 * np.log(np.pi * np.e / 2)           # differential entropy of |Z|
for a in [100, 1000]:
    asym = np.log(2) + 0.5 * (hplus + np.log(a / (2.0 / K)))
    report(f"ReLU asymptote log2 + (1/2)(h(|Z|) + log(a/Delta)), a={a}", asym, H_relu(a), rtol=2e-3)
amax = max([0.1 * 1.1 ** k for k in range(80)], key=H_tanh)
print(f"  tanh binned entropy peaks near a = {amax:.2f} at {H_tanh(amax):.3f} nats (rise then fall: 'fitting then compression')")

# ---------------------------------------------------------------------------
print("\n§6(c)  Gaussian channel and the two-class mixture")
d, sig = 3, 0.5
A = rng.standard_normal((d, d))
Sg = A @ A.T
# I(X;T) for Gaussian f(X) with cov S: h(T) - h(T|X)
I_formula = 0.5 * np.linalg.slogdet(np.eye(d) + Sg / sig ** 2)[1]
Ts = rng.multivariate_normal(np.zeros(d), Sg + sig ** 2 * np.eye(d), 400000)
Sigma_T = Sg + sig ** 2 * np.eye(d)
logp = -0.5 * np.einsum("ij,jk,ik->i", Ts, np.linalg.inv(Sigma_T), Ts) - 0.5 * np.linalg.slogdet(2 * np.pi * Sigma_T)[1]
hT_mc = -logp.mean()
I_mc = hT_mc - 0.5 * d * np.log(2 * np.pi * np.e * sig ** 2)
report("Gaussian: I = 1/2 log det(I + S/sigma^2)", I_formula, I_mc, rtol=0.01)

def binary_awgn_I(snr, rng, m=400000):
    # Y = +-1, log-odds L | Y=+1 ~ N(2 snr, 4 snr); I = log 2 - E log(1 + e^{-L})
    L = 2 * snr + 2 * np.sqrt(snr) * rng.standard_normal(m)
    return np.log(2) - np.mean(np.logaddexp(0, -L))

def mixture_I(mvec, W, sig, rng, m=400000):
    """MC of I(X;T) and I(T;Y): X=(Y,U), f(X)=Y m + W^{1/2} U, T=f(X)+sig eps."""
    d = len(mvec)
    ST = W + sig ** 2 * np.eye(d)
    Yv = rng.choice([-1.0, 1.0], m)
    T = Yv[:, None] * mvec + rng.multivariate_normal(np.zeros(d), ST, m)
    STi = np.linalg.inv(ST)
    ld = np.linalg.slogdet(2 * np.pi * ST)[1]
    lp = lambda c: -0.5 * np.einsum("ij,jk,ik->i", T - c, STi, T - c) - 0.5 * ld
    lpp, lpm = lp(mvec), lp(-mvec)
    lmix = np.logaddexp(lpp, lpm) - np.log(2)
    hT = -lmix.mean()
    I_XT = hT - 0.5 * d * np.log(2 * np.pi * np.e * sig ** 2)
    lpy = np.where(Yv > 0, lpp, lpm)
    I_TY = np.mean(lpy - lmix)
    return I_XT, I_TY

mvec = np.array([1.0, 0.0, 0.0])
W0 = np.diag([0.6, 1.0, 1.0])
print("  alpha (within-class scale) | I(X;T) MC | formula 1/2logdet(I+aW/s^2)+I(T;Y) | I(T;Y) MC | binary-AWGN I(T;Y) | Gaussian bound")
for alpha in [1.0, 0.3, 0.1, 0.01]:
    W = alpha * W0
    I_XT, I_TY = mixture_I(mvec, W, sig, rng)
    snr = mvec @ np.linalg.solve(W + sig ** 2 * np.eye(d), mvec)
    I_TY_f = binary_awgn_I(snr, rng)
    I_XT_f = 0.5 * np.linalg.slogdet(np.eye(d) + W / sig ** 2)[1] + I_TY_f
    bound = 0.5 * np.linalg.slogdet(np.eye(d) + (W + np.outer(mvec, mvec)) / sig ** 2)[1]
    print(f"   {alpha:5.2f} | {I_XT:.4f} | {I_XT_f:.4f} | {I_TY:.4f} | {I_TY_f:.4f} | {bound:.4f}")
    report(f"   decomposition I(X;T) = I(X;T|Y) + I(T;Y), alpha={alpha}", I_XT_f, I_XT, rtol=0.01, atol=3e-3)
    OK &= bool(I_XT <= bound + 1e-3)

# ---------------------------------------------------------------------------
print("\n§6(d)  IB distortion identity  E KL(p(y|x)||p(y|t)) = I(X;Y) - I(T;Y)")
nx, ny, nt = 6, 3, 4
pxy = rng.random((nx, ny)); pxy /= pxy.sum()
pt_x = rng.random((nx, nt)); pt_x /= pt_x.sum(1, keepdims=True)
px = pxy.sum(1); py = pxy.sum(0)
pxt = px[:, None] * pt_x
pt = pxt.sum(0)
pty = pt_x.T @ pxy                                # p(t,y) = sum_x p(t|x) p(x,y)
py_x = pxy / px[:, None]; py_t = pty / pt[:, None]
MI = lambda pj: np.sum(pj * np.log(pj / np.outer(pj.sum(1), pj.sum(0))))
EKL = np.sum(pxt[:, :, None] * py_x[:, None, :] * (np.log(py_x[:, None, :]) - np.log(py_t[None, :, :])))
report("E KL = I(X;Y) - I(T;Y)", MI(pxy) - MI(pty), EKL, rtol=1e-10)

print("\nALL OK" if OK else "\nSOME CHECKS FAILED")
raise SystemExit(0 if OK else 1)
