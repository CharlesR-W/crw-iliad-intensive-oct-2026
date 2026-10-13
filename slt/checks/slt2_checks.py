#!/usr/bin/env python3
"""Numerical checks of the closed forms in NB2's solutions (run: python checks/slt2_checks.py)."""
import numpy as np
from scipy.special import gamma

rng = np.random.default_rng(0)

# Exercise 2.1.4: volumes V(t) by Monte Carlo on [-1,1]^d (prior density 2^-d) against the closed forms.
t = 1e-3
N = 4_000_000
w = rng.uniform(-1, 1, N)
assert abs(np.mean(w**2 < t) - np.sqrt(t)) < 3e-3 * np.sqrt(t) + 2e-4
assert abs(np.mean(w**4 < t) - t**0.25) < 2e-3
a, b = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
V_ab = np.mean((a * b) ** 2 < t)
V_ab_closed = 0.5 * np.sqrt(t) * (-np.log(t)) + np.sqrt(t)
assert abs(V_ab - V_ab_closed) < 3e-4, (V_ab, V_ab_closed)
I = np.trapz(np.sqrt(1 - np.linspace(-1, 1, 200001) ** 4), np.linspace(-1, 1, 200001))
V_wv = np.mean(a**2 + b**4 < t)
assert abs(V_wv - 0.5 * I * t**0.75) < 3e-4, (V_wv, 0.5 * I * t**0.75)
print("2.1.4 volumes of w^2, w^4, (ab)^2, w^2+v^4: ok")

# Exercise 2.1.2: Zbar_n = c Gamma(lambda+1) n^-lambda for V = c t^lambda, by quadrature of n int e^{-nt} V(t) dt.
for lam, c in [(0.5, 1.0), (0.25, 1.0), (0.75, 0.5)]:
    n = 1000.0
    from scipy.integrate import quad
    Z = quad(lambda t: n * np.exp(-n * t) * c * t**lam, 0, np.inf)[0]      # n int e^{-nt} V(t) dt
    assert abs(Z / (c * gamma(lam + 1) * n ** (-lam)) - 1) < 1e-6
print("2.1.2 Laplace transform of c t^lambda: ok")

# Exercise 2.2: zeta for (ab)^2 with uniform prior equals 1/(2z+1)^2 (Monte Carlo at z = 0.3), and the pole lemma.
z = 0.3
assert abs(np.mean(((a * b) ** 2) ** z) - 1 / (2 * z + 1) ** 2) < 2e-3
u = np.linspace(1e-9, 60, 4_000_001)
for lam, m in [(0.5, 1), (0.5, 2), (0.25, 3)]:
    val = np.trapz(np.exp(-(z + lam) * u) * u ** (m - 1), u)
    assert abs(val / (np.math.factorial(m - 1) / (z + lam) ** m) - 1) < 1e-6
print("2.2.1-2 zeta of (ab)^2 and the pole lemma: ok")

# Exercise 2.3.2: n* for the two-basin example.
f = lambda n: 0.2 * n - 18 * np.log(n)
ns = np.arange(100, 2000)
nstar = ns[np.argmin(np.abs(f(ns)))]
assert 560 <= nstar <= 580, nstar
print(f"2.3.2 n* = {nstar}: ok")
