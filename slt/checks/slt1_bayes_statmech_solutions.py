#!/usr/bin/env python3
"""Checks of the 2026-10-14 corrections to NB1's worked solutions (run from slt/)."""
import numpy as np
import sympy as sp

# Ex 1.4.4: model N(w,1), data N(0, sigma2): L(w) - S = (w^2 + sigma2 - 1 - log sigma2)/2.
s2, w = sp.symbols("sigma2 w", positive=True)
L = sp.log(2 * sp.pi) / 2 + (s2 + w**2) / 2
S = sp.log(2 * sp.pi * sp.E * s2) / 2
assert sp.simplify(sp.expand_log(L - S - (w**2 + s2 - 1 - sp.log(s2)) / 2, force=True)) == 0
print("1.4.4 L(w)-S = (w^2+sigma^2-1-log sigma^2)/2: ok")

# Exact comparison: E[G_Bayes]-Lmin = log(1+1/n)/2, E[G_MLE]-Lmin = sigma2/2n; Bayes wins iff sigma2 > n log(1+1/n) < 1.
for n in [4, 10, 100]:
    thr = n * np.log1p(1 / n)
    assert thr < 1
    for sig in [thr - 1e-3, thr + 1e-3]:
        assert (0.5 * np.log1p(1 / n) < sig / (2 * n)) == (sig > thr)
print("1.4.4 Bayes beats MLE iff sigma^2 > n log(1+1/n) (0.893 at n=4): ok")

# Ex 1.1.3: F_min - log k <= F <= F_min, lower bound attained at equal masses.
F = np.array([2.0, 2.0, 2.0]); Ftot = -np.log(np.exp(-F).sum())
assert np.isclose(Ftot, F.min() - np.log(3))
print("1.1.3 lower bound attained at equal masses: ok")
