#!/usr/bin/env python3
"""Numerical checks of the closed forms in NB3's solutions (run: python checks/slt3_checks.py)."""
import numpy as np

# Exercise 3.1.2: lambda_eff = 1/4 + (s/2) (-log g)'(s) with g(s) = int exp(-u^4 - s u^2) du, and the s -> inf limit 1/2.
u = np.linspace(-6, 6, 1_200_001)
def g(s): return np.trapz(np.exp(-u**4 - s * u**2), u)
for s, lo, hi in [(0.0, 0.249, 0.251), (1e3, 0.49, 0.5)]:
    h = 1e-3 * max(s, 1)
    dlogg = (np.log(g(s + h)) - np.log(g(s - h))) / (2 * h) if s > 0 else (np.log(g(h)) - np.log(g(0))) / h
    lam_eff = 0.25 - 0.5 * s * dlogg
    assert lo <= lam_eff <= hi, (s, lam_eff)
# the same quantity from n E[K] directly, for delta = 0.1, n = 100 (s = 1)
w = np.linspace(-3, 3, 600_001); delta, n = 0.1, 100.0
K = w**4 + delta * w**2; p = np.exp(-n * K); p /= p.sum()
lam_direct = n * (p * K).sum()
s = delta * np.sqrt(n); h = 1e-3
lam_scaling = 0.25 - 0.5 * s * (np.log(g(s + h)) - np.log(g(s - h))) / (2 * h)
assert abs(lam_direct - lam_scaling) < 1e-3, (lam_direct, lam_scaling)
print("3.1.2 scaling form of lambda_eff and its limits: ok")

# Exercise 3.2.1: for V = c t^lambda, beta E_beta[nK] = lambda exactly (Gamma-function identity).
for lam in [0.25, 0.5, 0.75]:
    from scipy.integrate import quad
    nb = 3.0
    v = lambda t: lam * t ** (lam - 1)                      # V(t) = t^lambda
    Z = quad(lambda t: np.exp(-nb * t) * v(t), 0, np.inf)[0]
    EK = quad(lambda t: t * np.exp(-nb * t) * v(t), 0, np.inf)[0] / Z
    assert abs(nb * EK - lam) < 1e-6, (lam, nb * EK)
print("3.2.1 beta E[nK] = lambda for a pure power law: ok")

# Exercise 3.2.3: Gaussian localized estimator, 1/2 sum n beta h_i / (n beta h_i + gamma), against quadrature in 1-D.
w = np.linspace(-5, 5, 2_000_001)
for h, nb, gam in [(1.0, 10.0, 1.0), (0.1, 100.0, 30.0)]:
    K = 0.5 * h * w**2
    p = np.exp(-nb * K - 0.5 * gam * w**2); p /= p.sum()
    assert abs(nb * (p * K).sum() - 0.5 * nb * h / (nb * h + gam)) < 1e-6
print("3.2.3 localized Gaussian estimator: ok")
