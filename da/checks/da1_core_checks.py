#!/usr/bin/env python3
"""Numerical and symbolic checks for every formula stated in da1_core (DA1).

Run: python da/checks/da1_core_checks.py   (CPU, a few seconds). Prints one line per check
and exits non-zero on any failure. Conventions are DA1 §2's: summed objective
L(w; beta) = sum_i beta_i L_i(w), influence = d phi / d beta_i at beta = 1 (upweighting),
deletion = beta_i: 1 -> 0, so the first-order deletion prediction is -I.
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import sympy as sp
import torch
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------- §1 Shapley
# Croissant game: v(S) = 1 if butter or yeast premium. Players: flour, butter, yeast, sugar.
players = ["flour", "butter", "yeast", "sugar"]


def v(S):
    return 1.0 if ("butter" in S or "yeast" in S) else 0.0


def shapley(i):
    n = len(players)
    others = [p for p in players if p != i]
    tot = 0.0
    for k in range(n):
        for S in itertools.combinations(others, k):
            tot += (v(set(S) | {i}) - v(set(S))) / math.comb(n - 1, k)
    return tot / n


sh = {p: shapley(p) for p in players}
loo = {p: v(set(players)) - v(set(players) - {p}) for p in players}
check("Shapley croissant: butter = yeast = 1/2, flour = sugar = 0",
      abs(sh["butter"] - 0.5) < 1e-12 and abs(sh["yeast"] - 0.5) < 1e-12 and sh["flour"] == 0 == sh["sugar"], str(sh))
check("LOO croissant: every ingredient gets 0", all(x == 0 for x in loo.values()), str(loo))

# ----------------------------------------------------------------------------- §2 conventions
# Hampel: mean of (1-eps)F + eps delta_z has derivative z - T(F).
e, z, T = sp.symbols("epsilon z T")
check("Hampel mean influence z - T", sp.simplify(sp.diff((1 - e) * T + e * z, e) - (z - T)) == 0)

# Summed vs mean objective; Koh-Liang epsilon convention. Ridge-regularized logistic regression.
n, d, kappa = 40, 3, 0.5
X = rng.normal(size=(n, d))
y = (rng.random(n) < 1 / (1 + np.exp(-X @ np.array([1.5, -1.0, 0.5])))).astype(float)


def fit(beta, extra=None, reg=kappa, w0=None):
    """Newton for sum_i beta_i ell_i(w) + reg/2 |w|^2 (+ extra: (c, vec) adds c * ell(vec))."""
    w = np.zeros(d) if w0 is None else w0.copy()
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (beta * (p - y)) + reg * w
        H = (X.T * (beta * p * (1 - p))) @ X + reg * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.linalg.norm(step) < 1e-14:
            break
    return w


def grads(w):
    p = 1 / (1 + np.exp(-X @ w))
    return X * (p - y)[:, None], (X.T * (p * (1 - p))) @ X + kappa * np.eye(d)


ws = fit(np.ones(n))
G, H = grads(ws)
i = 7
I_param = -np.linalg.solve(H, G[i])
eps = 1e-6
b = np.ones(n); b[i] += eps
bm = np.ones(n); bm[i] -= eps
fd = (fit(b, w0=ws) - fit(bm, w0=ws)) / (2 * eps)
check("IF -H^{-1} grad L_i matches finite differences of retraining (summed objective)",
      np.allclose(fd, I_param, atol=1e-7), f"max err {np.abs(fd - I_param).max():.1e}")
# Mean objective (1/n) sum beta_i ell_i + reg/(2n)|w|^2 has the same argmin: same derivative.
# Koh-Liang: (1/n) sum ell_i + eps ell_i  ==  summed objective with beta_i = 1 + n eps.
I_KL = n * I_param                      # d w / d eps = n d w / d beta_i
check("Koh-Liang I_up = n * (our I); deletion eps=-1/n gives the same first-order change -I",
      np.allclose(I_KL * (-1 / n), -I_param))

# Damping = proximal term: argmin L(w;beta) + lam/2 |w - w*|^2 has derivative -(H+lam)^{-1} g_i.
lam = 3.0


def fit_prox(beta, lam, center):
    w = center.copy()
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (beta * (p - y)) + kappa * w + lam * (w - center)
        Hh = (X.T * (beta * p * (1 - p))) @ X + (kappa + lam) * np.eye(d)
        st = np.linalg.solve(Hh, g)
        w -= st
        if np.linalg.norm(st) < 1e-14:
            break
    return w


fdp = (fit_prox(b, lam, ws) - fit_prox(bm, lam, ws)) / (2 * eps)
check("damped IF -(H+lam)^{-1} g_i = derivative of the proximal response (at a critical point)",
      np.allclose(fdp, -np.linalg.solve(H + lam * np.eye(d), G[i]), atol=1e-7))

# LDS sign: training on S means beta_j = 0 for j not in S; first-order prediction
# phi(S) - phi(full) = -sum_{j not in S} I_j = sum_{j in S} I_j - sum_j I_j.
Ivec = rng.normal(size=10)
S = rng.random(10) < 0.5
check("LDS predictor: -sum_{j notin S} I_j = sum_{j in S} I_j - const",
      np.isclose(-Ivec[~S].sum(), Ivec[S].sum() - Ivec.sum()))

# ----------------------------------------------------------------------------- §3 OLS
n, p = 25, 4
Xo = rng.normal(size=(n, p)); Xo[0] *= 6          # one high-leverage point
yo = Xo @ rng.normal(size=p) + 0.5 * rng.normal(size=n)
A = Xo.T @ Xo
w_ols = np.linalg.solve(A, Xo.T @ yo)
r = yo - Xo @ w_ols
h = np.einsum("ij,jk,ik->i", Xo, np.linalg.inv(A), Xo)
check("leverages in [0,1] and sum to p", np.all((h >= 0) & (h <= 1)) and np.isclose(h.sum(), p),
      f"h_max={h.max():.3f}, sum={h.sum():.3f}")
ok_del, ok_beta, ok_if = True, True, True
for j in range(n):
    keep = np.arange(n) != j
    w_del = np.linalg.lstsq(Xo[keep], yo[keep], rcond=None)[0]
    pred = -np.linalg.solve(A, Xo[j]) * r[j] / (1 - h[j])
    ok_del &= np.allclose(w_del - w_ols, pred, atol=1e-10)
    for bj in (0.3, 1.7, -0.4):
        wt = np.ones(n); wt[j] = bj
        wb = np.linalg.solve((Xo.T * wt) @ Xo, (Xo.T * wt) @ yo)
        sm = -(1 - bj) * np.linalg.solve(A, Xo[j]) * r[j] / (1 - (1 - bj) * h[j])
        ok_beta &= np.allclose(wb - w_ols, sm, atol=1e-10)
    wt = np.ones(n); wt[j] += 1e-6
    wb = np.linalg.solve((Xo.T * wt) @ Xo, (Xo.T * wt) @ yo)
    ok_if &= np.allclose((wb - w_ols) / 1e-6, np.linalg.solve(A, Xo[j]) * r[j], atol=1e-5)
check("OLS exact deletion = -A^{-1} x_j r_j / (1 - h_j)", ok_del)
check("OLS general beta_j (Sherman-Morrison): -(1-b) A^{-1} x r / (1 - (1-b) h)", ok_beta)
check("OLS IF = A^{-1} x_j r_j", ok_if)
# Geometric-series form at beta_j: sum_k (1-b)^k h^{k-1}
bj, hj = 0.4, 0.3
series = sum((1 - bj) ** k * hj ** (k - 1) for k in range(1, 200))
check("geometric series sum_k (1-b)^k h^(k-1) = (1-b)/(1-(1-b)h)", np.isclose(series, (1 - bj) / (1 - (1 - bj) * hj)))
# h_j = 1 => deleting j loses identifiability: x_j is the only point with a component along e_p.
Xs = np.vstack([np.c_[rng.normal(size=(6, 2)), np.zeros(6)], [[0.3, -0.2, 1.0]]])
hs = np.einsum("ij,jk,ik->i", Xs, np.linalg.inv(Xs.T @ Xs), Xs)
check("h_j = 1 exactly when the retained design is rank-deficient",
      np.isclose(hs[-1], 1.0) and np.linalg.matrix_rank(Xs[:-1]) == 2, f"h_last={hs[-1]:.6f}")
# The direct derivation: delta = A^{-1} x_j c, c = -r_j/(1-h_j) (sympy, scalar unknown c).
c, hh, rr = sp.symbols("c h r")
sol = sp.solve(sp.Eq(c, hh * c - rr), c)[0]
check("deletion derivation: c = h c - r  =>  c = -r/(1-h)", sp.simplify(sol + rr / (1 - hh)) == 0)
# Leverage = Mahalanobis-type norm. No intercept: h = (1/n) x^T Shat^{-1} x, Shat = A/n uncentered.
Shat = A / n
check("h_j = (1/n) x_j^T (A/n)^{-1} x_j (uncentered second moment)",
      np.allclose(h, np.einsum("ij,jk,ik->i", Xo, np.linalg.inv(Shat), Xo) / n))
# With an intercept column: h_j = 1/n + (1/n) (x_j - xbar)^T C^{-1} (x_j - xbar), C the 1/n covariance.
Z = rng.normal(size=(n, 2)) @ np.array([[1, 0.4], [0, 0.7]]) + 1.5
Xi = np.c_[np.ones(n), Z]
hi = np.einsum("ij,jk,ik->i", Xi, np.linalg.inv(Xi.T @ Xi), Xi)
Zc = Z - Z.mean(0)
C = Zc.T @ Zc / n
check("with intercept: h_j = 1/n + (1/n) Mahalanobis^2 (centered, 1/n covariance)",
      np.allclose(hi, 1 / n + np.einsum("ij,jk,ik->i", Zc, np.linalg.inv(C), Zc) / n))

# ----------------------------------------------------------------------------- §4 GN counterexample
w = sp.symbols("w", real=True)
Lw = sp.Rational(1, 2) * (w**2 + 1) ** 2         # f = w^2, target -1, square loss
Hs = sp.diff(Lw, w, 2).subs(w, 0)
Gs = (sp.diff(w**2, w) ** 2).subs(w, 0)           # J^T H_y J with H_y = 1
check("GN counterexample f=w^2, target -1: at the strict minimum w=0, H=2, G=0",
      Hs == 2 and Gs == 0 and sp.diff(Lw, w).subs(w, 0) == 0)
# The general decomposition H = G + sum_c dl/df_c * Hess f_c, symbolically for a 2-parameter model.
a_, b_, t_ = sp.symbols("a b t", real=True)
f1 = a_ * b_ + sp.sin(a_)
Ls = sp.Rational(1, 2) * (f1 - t_) ** 2
Hfull = sp.hessian(Ls, (a_, b_))
J = sp.Matrix([[sp.diff(f1, a_), sp.diff(f1, b_)]])
Gfull = J.T * J
res = (f1 - t_) * sp.hessian(f1, (a_, b_))
check("H = G + (dl/df) Hess f (symbolic, 2-param model)", sp.simplify(Hfull - Gfull - res) == sp.zeros(2, 2))

# ----------------------------------------------------------------------------- §4 a*b model
zs = sp.symbols("z1:6", real=True)
a, bb, lamb = sp.symbols("a b lambda", positive=True)
nn = len(zs)
zbar = sum(zs) / nn
Lab = sum(sp.Rational(1, 2) * (zi - a * bb) ** 2 for zi in zs)
Hab = sp.hessian(Lab, (a, bb))
vvec = sp.Matrix([bb, a])
# at a minimizer: a*b = zbar.  Substitute b = zbar / a.
sub = {bb: zbar / a}
Hmin = sp.simplify(Hab.subs(sub))
check("a*b model: H = n v v^T at a minimizer", sp.simplify(Hmin - nn * (vvec * vvec.T).subs(sub)) == sp.zeros(2, 2))
gi = sp.Matrix([sp.diff(sp.Rational(1, 2) * (zs[0] - a * bb) ** 2, x) for x in (a, bb)]).subs(sub)
check("grad L_i = -(z_i - zbar) v", sp.simplify(gi + (zs[0] - zbar) * vvec.subs(sub)) == sp.zeros(2, 1))
Hl = Hmin + lamb * sp.eye(2)
for name, phi, expect in [
    ("ab", a * bb, (zs[0] - zbar) * (a**2 + bb**2) / (nn * (a**2 + bb**2) + lamb)),
    ("a^2", a**2, 2 * a * bb * (zs[0] - zbar) / (nn * (a**2 + bb**2) + lamb)),
]:
    gphi = sp.Matrix([sp.diff(phi, x) for x in (a, bb)]).subs(sub)
    Ilam = sp.simplify(-(gphi.T * Hl.inv() * gi)[0])
    check(f"a*b model: damped IF for phi={name}", sp.simplify(Ilam - expect.subs(sub)) == 0)
# pseudoinverse IF for phi = ab is (z_i - zbar)/n; LOO exact is -(z_i - zbar)/(n-1).
zn = rng.normal(size=8) + 2.0
nz = len(zn)
astar = 1.3
bstar = zn.mean() / astar
vv = np.array([bstar, astar])
Hn = nz * np.outer(vv, vv)
gin = -(zn[0] - zn.mean()) * vv
check("pseudoinverse IF for ab = (z_i - zbar)/n",
      np.isclose(-(vv @ np.linalg.pinv(Hn) @ gin), (zn[0] - zn.mean()) / nz))
check("exact LOO change of ab = -(z_i - zbar)/(n-1)",
      np.isclose(np.delete(zn, 0).mean() - zn.mean(), -(zn[0] - zn.mean()) / (nz - 1)))
# lambda -> 0 limit of the a^2 IF equals the derivative of warm-started gradient-flow retraining.
lim = 2 * astar * bstar * (zn[0] - zn.mean()) / (nz * (astar**2 + bstar**2))


def gf_endpoint(beta0):
    wts = np.ones(nz); wts[0] = beta0
    def rhs(t, w):
        u = w[0] * w[1]
        g = -np.sum(wts * (zn - u))
        return [-g * w[1], -g * w[0]]
    sol = solve_ivp(rhs, (0, 200), [astar, bstar], rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]


epsg = 1e-4
wp, wm = gf_endpoint(1 + epsg), gf_endpoint(1 - epsg)
fd_gf = (wp[0] ** 2 - wm[0] ** 2) / (2 * epsg)
check("a^2: lambda->0 damped IF = derivative of warm-started gradient-flow retraining",
      np.isclose(fd_gf, lim, rtol=1e-5), f"GF {fd_gf:.6f} vs limit {lim:.6f}")
check("gradient flow conserves a^2 - b^2", np.isclose(wp[0] ** 2 - wp[1] ** 2, astar**2 - bstar**2, rtol=1e-8))
# Minimizer dependence of the limit: balanced vs unbalanced.
zb = zn.mean()
lim_bal = 2 * zb * (zn[0] - zb) / (nz * 2 * zb)
lim_unb = 2 * zb * (zn[0] - zb) / (nz * (zb**2 + 1))
check("a^2 limit is minimizer dependent (balanced vs (zbar,1))", not np.isclose(lim_bal, lim_unb),
      f"{lim_bal:.4f} vs {lim_unb:.4f}")

# ----------------------------------------------------------------------------- §4 PBRF on a tiny MLP
torch.manual_seed(0)
Xt = torch.randn(12, 2)
yt = torch.sin(Xt[:, 0]) + 0.3 * Xt[:, 1]
shapes = [(5, 2), (5,), (1, 5), (1,)]
sizes = [int(np.prod(s)) for s in shapes]


def net(theta, Xin):
    W1, b1, W2, b2 = [q.reshape(s) for q, s in zip(torch.split(theta, sizes), shapes)]
    return (torch.tanh(Xin @ W1.T + b1) @ W2.T + b2).squeeze(-1)


theta_s = torch.randn(sum(sizes)) * 0.7            # a NON-converged checkpoint
f_s = net(theta_s, Xt).detach()
lam_p = 0.1
k = 3


def pbo(theta, beta_k):
    out = net(theta, Xt)
    breg = 0.5 * ((out - f_s) ** 2).sum()            # Bregman divergence of square loss
    pert = (beta_k - 1) * 0.5 * (out[k] - yt[k]) ** 2
    return breg + pert + 0.5 * lam_p * ((theta - theta_s) ** 2).sum()


def argmin_pbo(beta_k):
    th = theta_s.clone()
    for _ in range(50):
        g = torch.autograd.functional.jacobian(lambda t: pbo(t, beta_k), th)
        Hh = torch.autograd.functional.hessian(lambda t: pbo(t, beta_k), th)
        st = torch.linalg.solve(Hh, g)
        th = th - st
        if st.norm() < 1e-13:
            break
    return th


Jt = torch.autograd.functional.jacobian(lambda t: net(t, Xt), theta_s)
Gt = Jt.T @ Jt
g_k = torch.autograd.functional.jacobian(lambda t: 0.5 * (net(t, Xt)[k] - yt[k]) ** 2, theta_s)
pred_pbrf = -torch.linalg.solve(Gt + lam_p * torch.eye(len(theta_s)), g_k)
ep = 1e-5
fd_pbrf = (argmin_pbo(1 + ep) - argmin_pbo(1 - ep)) / (2 * ep)
grad_norm = torch.autograd.functional.jacobian(
    lambda t: 0.5 * ((net(t, Xt) - yt) ** 2).sum(), theta_s).norm().item()
check("PBRF derivative = -(G + lam I)^{-1} grad L_k at a non-converged checkpoint",
      torch.allclose(fd_pbrf, pred_pbrf, atol=1e-6), f"|grad L| at checkpoint = {grad_norm:.2f}")

# ----------------------------------------------------------------------------- §6 filters
eta, K = 0.1, 30
Hq = np.diag([0.0, 0.5, 3.0, 15.0])
g = rng.normal(size=4)
rr = np.zeros(4)
for _ in range(K):
    rr = rr - eta * (Hq @ rr) - eta * g             # d w_{t+1}/d beta_i, summed loss, frozen H and g
F = np.array([eta * K if s == 0 else (1 - (1 - eta * s) ** K) / s for s in np.diag(Hq)])
check("frozen-curvature GD response = -F(H) g with F = [1-(1-eta s)^K]/s, F(0) = eta K",
      np.allclose(rr, -F * g))
s_ = sp.symbols("s", positive=True)
eK = sp.symbols("T", positive=True)                 # T = eta K
Fsrc = (1 - sp.exp(-eK * s_)) / s_
check("SOURCE filter F(0) = eta K", sp.limit(Fsrc, s_, 0) == eK)
check("SOURCE at s = 1/(eta K): (1 - 1/e) eta K = 0.632 eta K",
      sp.simplify(Fsrc.subs(s_, 1 / eK) - (1 - sp.exp(-1)) * eK) == 0, f"{float(1 - math.exp(-1)):.4f}")
check("ridge 1/(s + 1/(eta K)) at s = 1/(eta K) is 0.5 eta K and F(0) = eta K",
      sp.simplify((1 / (s_ + 1 / eK)).subs(s_, 1 / eK) - eK / 2) == 0 and sp.simplify((1 / (s_ + 1 / eK)).subs(s_, 0) - eK) == 0)
# GD filter -> 1/s as K -> infinity for 0 < eta s < 2, and blows up for eta s > 2.
check("GD filter -> 1/s for 0 < eta s < 2", np.isclose((1 - (1 - 0.1 * 5) ** 2000) / 5, 1 / 5))

# Bayesian linear regression: BIF = -Cov(NLL_j, w) = V x_j r_j / sigma^2 = (A + sigma^2/tau^2)^{-1} x_j r_j.
n2, p2, sig, tau = 30, 3, 0.7, 1.3
Xb = rng.normal(size=(n2, p2))
yb = Xb @ rng.normal(size=p2) + sig * rng.normal(size=n2)
Ab = Xb.T @ Xb
V = np.linalg.inv(np.eye(p2) / tau**2 + Ab / sig**2)
mu = V @ Xb.T @ yb / sig**2
j = 4
rj = yb[j] - Xb[j] @ mu
u = rng.multivariate_normal(np.zeros(p2), V, size=2_000_000)
Lj = (rj - u @ Xb[j]) ** 2 / (2 * sig**2)
cov_mc = ((Lj - Lj.mean())[:, None] * u).mean(0)
bif = np.linalg.solve(Ab + sig**2 / tau**2 * np.eye(p2), Xb[j] * rj)
check("BLR: -Cov(NLL_j, w) = V x_j r_j / sigma^2 (Monte Carlo)", np.allclose(-cov_mc, V @ Xb[j] * rj / sig**2, atol=3e-3),
      f"max err {np.abs(-cov_mc - V @ Xb[j] * rj / sig**2).max():.1e}")
check("BLR: V x r / sigma^2 = (A + sigma^2/tau^2 I)^{-1} x r  (damping sigma^2/tau^2 in X^T X units)",
      np.allclose(V @ Xb[j] * rj / sig**2, bif))
Hlik = Ab / sig**2
check("BLR in likelihood-Hessian units: (H + tau^{-2} I)^{-1} grad(-NLL_j), damping tau^{-2}",
      np.allclose(np.linalg.solve(Hlik + np.eye(p2) / tau**2, Xb[j] * rj / sig**2), bif))
# BIF of the weight on example j, exactly: d mu / d beta_j by finite differences of the conjugate posterior.


def post_mean(beta):
    Vb = np.linalg.inv(np.eye(p2) / tau**2 + (Xb.T * beta) @ Xb / sig**2)
    return Vb @ (Xb.T * beta) @ yb / sig**2


bp = np.ones(n2); bp[j] += 1e-6
check("BLR: d E[w] / d beta_j = (A + sigma^2/tau^2)^{-1} x_j r_j exactly",
      np.allclose((post_mean(bp) - mu) / 1e-6, bif, atol=1e-5))

print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
