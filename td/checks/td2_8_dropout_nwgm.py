#!/usr/bin/env python3
"""Check for td2_8 §1: the NWGM identity and the small-spread gap E - NWGM.

1. NWGM = sigma(E[S]) exactly (exact enumeration over 2^n masks).
2. Leading order in the spread s^2 of S:
      E - NWGM = (1/2) sigma''(mu) s^2 + O(s^3)    and    V = sigma'(mu)^2 s^2 + O(s^3),
   hence E - NWGM ~ V (1 - 2E) / (2 E (1 - E)).
   Baldi & Sadowski (NeurIPS 2013) eq. (20) gives V (1 - 2E) / (1 - 2V); in the
   small-spread limit the ratio exact/eq.(20) -> 1 / (2 E (1 - E)) >= 2.
"""
import itertools

import numpy as np
import sympy as sp

# --- symbolic: leading-order gap for a symmetric two-point spread S = mu +/- s
mu, s = sp.symbols("mu s", real=True)
sig = lambda x: 1 / (1 + sp.exp(-x))
E = (sig(mu + s) + sig(mu - s)) / 2
V = (sig(mu + s) - E) ** 2 / 2 + (sig(mu - s) - E) ** 2 / 2
gap = E - sig(mu)                     # NWGM = sigma(E S) = sigma(mu)
lead_gap = sp.simplify(sp.series(gap, s, 0, 3).removeO())
lead_V = sp.simplify(sp.series(V, s, 0, 3).removeO())
sg = sig(mu)
assert sp.simplify(lead_gap - sp.diff(sg, mu, 2) * s**2 / 2) == 0
assert sp.simplify(lead_V - sp.diff(sg, mu) ** 2 * s**2) == 0
ratio = sp.simplify(lead_gap / (lead_V * (1 - 2 * sg)))
assert sp.simplify(ratio - 1 / (2 * sg * (1 - sg))) == 0
print("symbolic: E - NWGM ~ V(1-2E)/(2E(1-E)) at leading order  OK")

# --- numeric: exact enumeration
rng = np.random.default_rng(0)
n = 12
masks = np.array(list(itertools.product([0, 1], repeat=n)))
sigma = lambda x: 1 / (1 + np.exp(-x))


def stats(w, I, p, b=0.0):
    P = np.prod(np.where(masks == 1, p, 1 - p), axis=1)
    S = masks @ (w * I) + b
    O = sigma(S)
    E = P @ O
    V = P @ (O - E) ** 2
    G, Gc = np.exp(P @ np.log(O)), np.exp(P @ np.log(1 - O))
    return E, V, G / (G + Gc), sigma(P @ S)


E_, V_, N_, sES = stats(rng.normal(0, 1, n), rng.normal(0, 1, n), rng.uniform(0.3, 0.9, n))
assert abs(N_ - sES) < 1e-12
print(f"NWGM - sigma(E S) = {N_ - sES:.1e}  OK")

w, I, p = rng.normal(0, 0.05, n), rng.normal(0, 1, n), np.full(n, 0.5)
for b in [0.3, 1.0, 2.0, 3.0]:
    E_, V_, N_, _ = stats(w, I, p, b)
    mine = V_ * (1 - 2 * E_) / (2 * E_ * (1 - E_))
    bs20 = V_ * (1 - 2 * E_) / (1 - 2 * V_)
    print(f"b={b}: exact {E_ - N_:+.4e}  leading-order {mine:+.4e} (ratio {(E_ - N_) / mine:.3f})  "
          f"eq.(20) {bs20:+.4e} (exact/eq20 {(E_ - N_) / bs20:.2f}, 1/(2E(1-E)) {1 / (2 * E_ * (1 - E_)):.2f})")
    assert abs((E_ - N_) / mine - 1) < 0.05
