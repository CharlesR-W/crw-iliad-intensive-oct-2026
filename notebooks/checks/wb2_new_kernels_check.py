"""Numerical checks for WB2 1.E (kernels and the kernel trick).

Run: /home/crw/Programming/CellsDevelopmental/.venv/bin/python notebooks/checks/wb2_new_kernels_check.py
Every check asserts; the script prints one line per check.
"""
from itertools import product
from math import comb, factorial

import numpy as np

rng = np.random.default_rng(0)


def ok(name, cond, extra=""):
    assert cond, name
    print(f"ok  {name}  {extra}")


# ---------------------------------------------------------------- 1.E.1 dual ridge
for n, D in [(7, 20), (20, 7)]:
    Phi = rng.normal(size=(n, D))
    y = rng.normal(size=n)
    lam = 0.3
    a = n * lam
    K = Phi @ Phi.T
    # push-through identity
    lhs = np.linalg.solve(Phi.T @ Phi + a * np.eye(D), Phi.T)
    rhs = Phi.T @ np.linalg.inv(K + a * np.eye(n))
    ok(f"push-through identity n={n} D={D}", np.allclose(lhs, rhs))
    w_primal = lhs @ y
    alpha = np.linalg.solve(K + a * np.eye(n), y)
    ok(f"w = Phi^T alpha n={n} D={D}", np.allclose(w_primal, Phi.T @ alpha))
    phix = rng.normal(size=D)
    kx = Phi @ phix
    ok("prediction k_x^T (K+n lam)^-1 y", np.isclose(w_primal @ phix, kx @ alpha))
    # objective in alpha: (1/n)||y-K alpha||^2 + lam alpha^T K alpha, stationary at alpha
    grad = -2 / n * K @ (y - K @ alpha) + 2 * lam * K @ alpha
    ok("alpha stationary for dual objective", np.allclose(grad, 0))
    # RKHS norm of fitted function = ||w||^2 = alpha^T K alpha
    ok("||w||^2 = alpha^T K alpha", np.isclose(w_primal @ w_primal, alpha @ K @ alpha))
    # spectral filter of fitted values
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    yhat = Phi @ w_primal
    yhat_filter = U @ (s**2 / (s**2 + a) * (U.T @ y))
    ok("fitted values = sum s^2/(s^2+n lam) (u^T y) u", np.allclose(yhat, yhat_filter))
    norm_filter = np.sum(s**2 * (U.T @ y) ** 2 / (s**2 + a) ** 2)
    ok("||f||_H^2 = sum s^2 (u^T y)^2/(s^2+a)^2", np.isclose(norm_filter, w_primal @ w_primal))

# representer: adding an orthogonal component never helps
n, D = 6, 15
Phi = rng.normal(size=(n, D)); y = rng.normal(size=n); lam = 0.1
obj = lambda w: np.mean((y - Phi @ w) ** 2) + lam * w @ w
w_hat = np.linalg.solve(Phi.T @ Phi + n * lam * np.eye(D), Phi.T @ y)
P_perp = np.eye(D) - np.linalg.pinv(Phi) @ Phi
worse = all(obj(w_hat + P_perp @ rng.normal(size=D)) > obj(w_hat) for _ in range(100))
ok("orthogonal component only increases objective", worse)

# ridgeless limit = min-norm interpolant, norm y^T K^-1 y
n, D = 8, 30
Phi = rng.normal(size=(n, D)); y = rng.normal(size=n); K = Phi @ Phi.T
w0 = np.linalg.pinv(Phi) @ y
w_small = Phi.T @ np.linalg.solve(K + 1e-9 * np.eye(n), y)
ok("ridgeless limit = Phi^+ y", np.allclose(w0, w_small, atol=1e-6))
ok("min norm = y^T K^-1 y", np.isclose(w0 @ w0, y @ np.linalg.solve(K, y)))
ok("interpolates", np.allclose(Phi @ w0, y))

# expected ridgeless norm: E||w0||^2 = ||P_row w*||^2 + sig^2 Tr K^-1
n, D, sig = 8, 30, 0.4
Phi = rng.normal(size=(n, D)); K = Phi @ Phi.T; wstar = rng.normal(size=D)
Kinv = np.linalg.inv(K)
f = Phi @ wstar
P = np.linalg.pinv(Phi) @ Phi
exact = f @ Kinv @ f + sig**2 * np.trace(Kinv)
ok("f*^T K^-1 f* = ||P_row w*||^2", np.isclose(f @ Kinv @ f, np.sum((P @ wstar) ** 2)))
eps = sig * rng.normal(size=(200000, n))
Y = f + eps
mc = np.mean(np.einsum("ti,ij,tj->t", Y, Kinv, Y))
ok("E y^T K^-1 y = f^T K^-1 f + sig^2 Tr K^-1", abs(mc / exact - 1) < 0.02, f"{mc:.3f} vs {exact:.3f}")

# constant 1 in Gaussian RKHS: w_{2k}^2 = (2k)!/(4^k k!^2) ~ 1/sqrt(pi k), partial sums of e^{x^2/2l^2}
ell, x = 0.8, 0.5
J = 60
w = np.zeros(J)
for k in range(J // 2):
    w[2 * k] = np.sqrt(float(factorial(2 * k))) / (2**k * factorial(k))
phi1 = np.array([np.exp(-x**2 / (2 * ell**2)) * (x / ell) ** j / np.sqrt(float(factorial(j))) for j in range(J)])
ok("weights w_{2k} reproduce the constant 1", np.isclose(w @ phi1, 1.0))
ok("w_{2k}^2 ~ 1/sqrt(pi k)", np.isclose(w[2 * 29] ** 2 * np.sqrt(np.pi * 29), 1, atol=0.01))

# ---------------------------------------------------------------- 1.E.2 kernels
# polynomial kernel explicit features
d, m = 3, 3
idx = [j for j in product(range(m + 1), repeat=d) if sum(j) <= m]
ok("poly feature count = C(d+m, m)", len(idx) == comb(d + m, m), f"{len(idx)}")


def poly_phi(x):
    out = []
    for j in idx:
        j0 = m - sum(j)
        coef = factorial(m) / (factorial(j0) * np.prod([factorial(t) for t in j]))
        out.append(np.sqrt(coef) * np.prod(x ** np.array(j)))
    return np.array(out)


x, xp = rng.normal(size=d), rng.normal(size=d)
ok("poly features reproduce (1+x.x')^m", np.isclose(poly_phi(x) @ poly_phi(xp), (1 + x @ xp) ** m))
print("    feature counts:", {(dd, mm): f"{comb(dd + mm, mm):.2e}" for dd, mm in [(784, 3), (784, 5), (1000, 5)]})

# Gaussian kernel 1-D Taylor features
ell = 0.7
x, xp = 0.9, -0.4
J = 40
phi = lambda t: np.array([np.exp(-t**2 / (2 * ell**2)) * (t / ell) ** j / np.sqrt(float(factorial(j))) for j in range(J)])
ok("Gaussian 1-D Taylor features", np.isclose(phi(x) @ phi(xp), np.exp(-(x - xp) ** 2 / (2 * ell**2))))

# -||x-x'||^2 is not PD; min(x,x') is PD
r = 1.3
G = np.array([[0, -r**2], [-r**2, 0]])
ok("-||x-x'||^2 Gram has negative eigenvalue", np.linalg.eigvalsh(G).min() < 0)
t = np.sort(rng.uniform(0, 5, size=30))
ok("min(x,x') Gram PSD", np.linalg.eigvalsh(np.minimum.outer(t, t)).min() > -1e-10)
# feature map 1_[0,x] in L2: integral of product = min
grid = np.linspace(0, 5, 200001)
ok("int 1[0,x]1[0,x'] = min(x,x')", np.isclose(np.mean((grid <= 1.7) * (grid <= 3.2)) * 5, 1.7, atol=1e-4))

# Gaussian kernel Gram strictly PD for distinct points
X = rng.normal(size=(40, 2))
sq = ((X[:, None] - X[None]) ** 2).sum(-1)
Kg = np.exp(-sq / 2)
ok("Gaussian Gram strictly PD (distinct points)", np.linalg.eigvalsh(Kg).min() > 0, f"min eig {np.linalg.eigvalsh(Kg).min():.2e}")

# random Fourier features
d, ell = 3, 1.2
x, xp = rng.normal(size=d) * 0.6, rng.normal(size=d) * 0.6
kxx = np.exp(-np.sum((x - xp) ** 2) / (2 * ell**2))
M = 2_000_000
W = rng.normal(size=(M, d)) / ell
b = rng.uniform(0, 2 * np.pi, size=M)
terms = 2 * np.cos(W @ x + b) * np.cos(W @ xp + b)
ok("E[2cos cos] = Gaussian kernel", abs(terms.mean() - kxx) < 5 / np.sqrt(M), f"{terms.mean():.4f} vs {kxx:.4f}")
var_pred = 1 + kxx**4 / 2 - kxx**2
ok("Var[2cos cos] = 1 + k^4/2 - k^2", abs(terms.var() - var_pred) < 0.01, f"{terms.var():.4f} vs {var_pred:.4f}")
terms0 = 2 * np.cos(W @ x + b) ** 2
ok("variance 1/2 at x=x'", abs(terms0.var() - 0.5) < 0.01)

# RFF ridge approaches kernel ridge
n, d, ell, lam = 60, 2, 1.0, 1e-3
X = rng.uniform(-2, 2, size=(n, d)); y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * rng.normal(size=n)
Xt = rng.uniform(-2, 2, size=(200, d))
kern = lambda A, B: np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (2 * ell**2))
f_krr = kern(Xt, X) @ np.linalg.solve(kern(X, X) + n * lam * np.eye(n), y)
errs = []
for m in [50, 500, 5000, 50000]:
    W = rng.normal(size=(m, d)) / ell; b = rng.uniform(0, 2 * np.pi, size=m)
    z = lambda A: np.sqrt(2 / m) * np.cos(A @ W.T + b)
    Z = z(X)
    w = np.linalg.solve(Z.T @ Z + n * lam * np.eye(m), Z.T @ y) if m < n * 10 else Z.T @ np.linalg.solve(Z @ Z.T + n * lam * np.eye(n), y)
    errs.append(np.sqrt(np.mean((z(Xt) @ w - f_krr) ** 2)))
ok("RFF ridge -> KRR as m grows", errs[-1] < errs[0] and errs[-1] < 0.02, str([f"{e:.3g}" for e in errs]))

# ---------------------------------------------------------------- 1.E.3 GP
n, D = 10, 25
Phi = rng.normal(size=(n, D)); y = rng.normal(size=n)
tau2, sig2 = 0.8, 0.3
K = Phi @ Phi.T
phix = rng.normal(size=D); kx = Phi @ phix; kxx = phix @ phix
# function space (prior f ~ GP(0, tau2 k))
mean_fs = tau2 * kx @ np.linalg.solve(tau2 * K + sig2 * np.eye(n), y)
var_fs = tau2 * kxx - tau2**2 * kx @ np.linalg.solve(tau2 * K + sig2 * np.eye(n), kx)
# weight space: w ~ N(0, tau2 I)
Sw = np.linalg.inv(Phi.T @ Phi / sig2 + np.eye(D) / tau2)
mw = Sw @ Phi.T @ y / sig2
ok("GP mean: function space = weight space", np.isclose(mean_fs, phix @ mw))
ok("GP var: function space = weight space", np.isclose(var_fs, phix @ Sw @ phix))
lam = sig2 / (n * tau2)
krr = kx @ np.linalg.solve(K + n * lam * np.eye(n), y)
ok("GP mean = KRR with n lam = sig2/tau2", np.isclose(mean_fs, krr))
var_alt = tau2 * (kxx - kx @ np.linalg.solve(K + sig2 / tau2 * np.eye(n), kx))
ok("GP var = tau2 (k(x,x) - k_x^T (K + sig2/tau2 I)^-1 k_x)", np.isclose(var_fs, var_alt))
# posterior variance independent of y: trivially; at training point with sig2 -> 0 -> 0
x_tr = Phi[3]
kx_tr = Phi @ x_tr
v0 = x_tr @ x_tr - kx_tr @ np.linalg.solve(K + 1e-10 * np.eye(n), kx_tr)
ok("noiseless posterior var at training point = 0", abs(v0) < 1e-6)
# noiseless variance = sup over unit ball of (g(x) - ghat(x))^2
c = np.linalg.solve(K, kx)
resid = phix - Phi.T @ c
v_noiseless = kxx - kx @ np.linalg.solve(K, kx)
ok("power function = ||phi(x) - Phi^T c||^2", np.isclose(resid @ resid, v_noiseless))
best = 0
for _ in range(20000):
    w = rng.normal(size=D); w /= np.linalg.norm(w)
    g_vals = Phi @ w
    ghat_x = kx @ np.linalg.solve(K, g_vals)
    best = max(best, (phix @ w - ghat_x) ** 2)
w_star = resid / np.linalg.norm(resid)
val_star = (phix @ w_star - kx @ np.linalg.solve(K, Phi @ w_star)) ** 2
ok("sup attained at w = resid/||resid||", np.isclose(val_star, v_noiseless) and best <= v_noiseless + 1e-9)

print("all checks passed")
