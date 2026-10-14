#!/usr/bin/env python3
"""Checks behind the NB3 illustrations and the reworded hyperparameter text (2026-10-14).

Run from slt/:  python checks/slt3_effective_temperature_figures.py
"""
import numpy as np

# Figure 1(a): lambda_eff(n; delta) for K = w^4 + delta w^2 (flat prior on R) is a function of
# s = delta sqrt(n), equals 1/4 at delta = 0 for every n, increases with delta, tends to 1/2.
u = np.linspace(-6, 6, 400_001)
def lam_eff_s(s):
    e = u**4 + s * u**2; p = np.exp(-e); return (p * e).sum() / p.sum()
vals = [lam_eff_s(s) for s in [0, 0.3, 1, 3, 10, 100, 1000]]
assert abs(vals[0] - 0.25) < 1e-6 and all(np.diff(vals) > 0) and 0.49 < vals[-1] < 0.5, vals
# the same from n E[K] on a w-grid, delta = 0.05, n = 400 (s = 1)
w = np.linspace(-3, 3, 600_001); d, n = 0.05, 400.0
K = w**4 + d * w**2; p = np.exp(-n * K); p /= p.sum()
assert abs(n * (p * K).sum() - lam_eff_s(1.0)) < 1e-4
print("fig 1(a): lambda_eff depends on delta sqrt(n) only, 1/4 at delta=0, rises to 1/2: ok")

# Figure 1(b): K_n = a w^4/2 - u w^2 (Exercise 3.1.4 at delta = 0) is a double well iff u > 0, minima at w^2 = u/a.
for a_, u_ in [(1.0, 0.03), (1.1, 0.06)]:
    ww = np.linspace(0, 1, 1_000_001); Kn = 0.5 * a_ * ww**4 - u_ * ww**2
    assert abs(ww[np.argmin(Kn)] ** 2 - u_ / a_) < 1e-4
print("fig 1(b): double-well minima at w^2 = u_n/a_n: ok")

# Figure 2(c): in 1-D, K = |w|^k gives V(t) = 2 t^(1/k) (uniform Lebesgue), so lambda = 1/slope;
# the two terms of w^4 + delta w^2 are equal at w^2 = delta, where K = 2 delta^2.
for k in [2, 4]:
    t1, t2 = 1e-6, 1e-4
    V = lambda t: 2 * t ** (1 / k)
    assert abs(np.log(V(t2) / V(t1)) / np.log(t2 / t1) - 1 / k) < 1e-12
print("fig 2(c): lambda = 1/(log-log slope) in 1-D: ok")

# Figure 3: E_beta[nK] = lambda/beta for V = c t^lambda (exact; see slt3_checks 3.3.1), so the mean of n L_n is
# linear in 1/beta with slope lambda, equals nL(w*) + lambda at beta = 1 and nL(w*) + lambda log n at beta = 1/log n.
from scipy.integrate import quad
lam, n = 0.75, 1000.0
for beta in [1.0, 1 / np.log(n)]:
    nb = n * beta
    Z = quad(lambda t: np.exp(-nb * t) * lam * t ** (lam - 1), 0, np.inf)[0]
    EnK = n * quad(lambda t: t * np.exp(-nb * t) * lam * t ** (lam - 1), 0, np.inf)[0] / Z
    assert abs(EnK - lam / beta) < 1e-6
print("fig 3: E_beta[nK] = lambda/beta, = lambda at beta=1 and lambda log n at beta=1/log n: ok")

# Figure 4: K = (w1 w2)^2, n beta = 200, localizer gamma/2 |w|^2. A tighter tether lowers
# lambda_hat = n beta E[K] (exact on a grid); the loose-tether value is below lambda = 1/2 (m = 2 log correction).
g = np.linspace(-4, 4, 2001); W1, W2 = np.meshgrid(g, g)
K = (W1 * W2) ** 2; nb = 200.0
lh = {}
for gam in [1.0, 25.0]:
    logp = -nb * K - 0.5 * gam * (W1**2 + W2**2); p = np.exp(logp - logp.max()); p /= p.sum()
    lh[gam] = nb * (p * K).sum()
assert lh[25.0] < lh[1.0] < 0.5, lh
print(f"fig 4: lambda_hat(gamma=1) = {lh[1.0]:.3f} > lambda_hat(gamma=25) = {lh[25.0]:.3f}: ok")

# Hyperparameters: the linearised update w <- w - (eps/2)(n beta h + gamma) w is stable iff eps (n beta h + gamma) < 4.
for c in [3.9, 4.1]:
    ww = 1.0
    for _ in range(2000): ww = ww - 0.5 * c * ww
    assert (abs(ww) < 1) == (c < 4)
print("step-size stability: eps (n beta h + gamma) < 4: ok")
