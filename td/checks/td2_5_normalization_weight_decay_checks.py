#!/usr/bin/env python3
"""Checks for the closed-form results stated in td2_5_normalization_weight_decay.

Run: python td/checks/td2_5_normalization_weight_decay_checks.py   (a few seconds)
Each check prints OK or raises.
"""
import math

import numpy as np
import sympy as sp

e, l, G, s, C = sp.symbols("eta lambda G s C", positive=True)
a = (1 - e * l) ** 2

# 5.2: SGD+WD on a scale-invariant loss, s = r^2:  s' = a s + eta^2 G^2 / s
s_star = sp.sqrt(e**2 * G**2 / (1 - a))
assert sp.simplify(s_star**2 - e * G**2 / (l * (2 - e * l))) == 0
tan2 = sp.simplify((e**2 * G**2 / s_star) / ((1 - e * l) ** 2 * s_star))
assert sp.simplify(tan2 - e * l * (2 - e * l) / (1 - e * l) ** 2) == 0
assert sp.simplify(1 / (1 + tan2) - (1 - e * l) ** 2) == 0          # cos(theta*) = 1 - eta*lambda
slope = sp.diff(a * s + e**2 * G**2 / s, s).subs(s, s_star)
assert sp.simplify(slope - (2 * (1 - e * l) ** 2 - 1)) == 0          # ~ 1 - 4 eta lambda
assert sp.series(sp.acos(1 - sp.Symbol("x")), sp.Symbol("x"), 0, 2).removeO().has(sp.sqrt(2))
x = sp.Symbol("x", positive=True)
assert sp.limit(sp.acos(1 - x) / sp.sqrt(2 * x), x, 0) == 1
# equilibrium effective LR eta/((1-eta*lam) r*^2) * G = tan(theta*)
eff = e / ((1 - e * l) * s_star) * G
assert sp.simplify(eff**2 - tan2) == 0
print("OK  SGD+WD fixed point, cos(theta*) = 1 - eta*lam, slope 2(1-eta lam)^2 - 1, eff LR")

# 5.2: only eta*lam matters: with s = r^2/eta, s' = a s + G^2/s and eta_eff = 1/((1-eta lam) s)
r2 = sp.Symbol("r2", positive=True)
lhs = (a * r2 + e**2 * G**2 / r2) / e
rhs = a * (r2 / e) + G**2 / (r2 / e)
assert sp.simplify(lhs - rhs) == 0
print("OK  rescaled norm recursion depends on eta*lam only")

# 5.2: AdamW (beta1 = 0): s' = a s + eta^2 C
s2 = e**2 * C / (1 - a)
assert sp.simplify(s2 - e * C / (l * (2 - e * l))) == 0
assert sp.simplify(1 / (1 + e**2 * C / ((1 - e * l) ** 2 * s2)) - (1 - e * l) ** 2) == 0
print("OK  AdamW(beta1=0) equilibrium norm and cos(theta*) = 1 - eta*lam")

# 5.2: Li-Arora (no momentum): w~_t = w_t (1-eta lam)^(-t) gives LR eta (1-eta lam)^(-2t-1)
t = sp.Symbol("t", integer=True)
q = 1 - e * l
assert sp.simplify(e * q ** (-(t + 1)) * q ** (-t) - e * q ** (-2 * t - 1)) == 0
print("OK  Li-Arora schedule exponent -2t-1")

# 5.3: Adam frozen-preconditioner threshold at beta1 = 0.9, and time for v to decay by F^2
b1 = 0.9
assert abs((2 + 2 * b1) / (1 - b1) - 38) < 1e-12
k999 = 2 * math.log(2) / -math.log(0.999)
k9 = 2 * math.log(2) / -math.log(0.9)
assert abs(k999 - 1385.6) < 0.5 and abs(k9 - 13.16) < 0.05
print(f"OK  38/eta threshold; doubling time of eta*h/sqrt(v): {k999:.0f} steps (beta2=.999), {k9:.1f} (beta2=.9)")

# 5.3: sign descent on the output layer, batch of one example
rng = np.random.default_rng(0)
K, d, eta = 7, 5, 1e-3
for _ in range(100):
    W, b, h = rng.standard_normal((K, d)), rng.standard_normal(K), rng.standard_normal(d)
    y = rng.integers(K)
    z = W @ h + b
    p = np.exp(z - z.max()); p /= p.sum()
    gz = p.copy(); gz[y] -= 1
    W2 = W - eta * np.sign(np.outer(gz, h)); b2 = b - eta * np.sign(gz)
    z2 = W2 @ h + b2
    dz = z2 - z
    s_ = np.sign(gz)
    assert np.allclose(dz, -eta * s_ * (np.abs(h).sum() + 1))
    assert np.isclose(dz.mean(), -eta * (np.abs(h).sum() + 1) * (K - 2) / K)
    dlogZ = np.log(np.exp(z2).sum()) - np.log(np.exp(z).sum())
    pred = -eta * (np.abs(h).sum() + 1) * (1 - 2 * p[y])
    assert abs(dlogZ - pred) < 50 * eta**2 * (np.abs(h).sum() + 1) ** 2
    # SGD conserves the column sums of W and the sum of b
    Wg = np.outer(gz, h)
    assert np.allclose(Wg.sum(0), 0) and np.isclose(gz.sum(), 0)
print("OK  sign-descent logit drift formulas; SGD conserves the shift direction")

# 5.3: QK-layernorm bound: |LN(q).LN(k)|/sqrt(dh) <= sqrt(dh)
dh = 64
for _ in range(1000):
    qv, kv = rng.standard_normal(dh) * 10, rng.standard_normal(dh) * 3
    ln = lambda v: (v - v.mean()) / v.std()
    assert abs(ln(qv) @ ln(kv)) / math.sqrt(dh) <= math.sqrt(dh) + 1e-9
    assert np.isclose(np.linalg.norm(ln(qv)), math.sqrt(dh))
print("OK  QK-layernorm logit bound sqrt(d_h)")

# 5.4: (L0,L1) descent-lemma constant: ((e^u - 1)/u - 1)/u <= e - 2 on (0, 1], increasing
u = np.linspace(1e-6, 1, 100001)
phi = ((np.exp(u) - 1) / u - 1) / u
assert np.all(np.diff(phi) > 0) and abs(phi[-1] - (math.e - 2)) < 1e-9 and phi[0] > 0.4999
kappa = 2 * (math.e - 2)
assert kappa > 1          # so h*|g| = |g|/(kappa(L0+L1|g|)) < 1/L1
print(f"OK  descent-lemma constant e-2; kappa = 2(e-2) = {kappa:.4f} > 1")

# 5.4: cosh is (1,1)-smooth: cosh x <= 1 + |sinh x|
xs = np.linspace(-30, 30, 200001)
assert np.all(np.cosh(xs) <= 1 + np.abs(np.sinh(xs)) + 1e-9 * np.cosh(xs))
# Taylor remainder bound checked on random points for cosh with |d| <= 1
for _ in range(20000):
    x0, dd = rng.uniform(-20, 20), rng.uniform(-1, 1)
    rem = math.cosh(x0 + dd) - math.cosh(x0) - math.sinh(x0) * dd
    assert rem <= (math.e - 2) * (1 + abs(math.sinh(x0))) * dd * dd * (1 + 1e-12) + 1e-12
print("OK  cosh is (L0,L1)=(1,1)-smooth and obeys the descent lemma with constant e-2")

# 5.4: fixed-step GD on cosh from x0 must have h sinh(x0) <= 2 x0 to avoid |x1| > x0
for x0 in [3.0, 6.0, 10.0]:
    hmax = 2 * x0 / math.sinh(x0)
    x1 = x0 - 1.01 * hmax * math.sinh(x0)
    assert abs(x1) > x0
print("OK  GD overshoot condition on cosh")

# 5.4: Adam (beta1 = 0) per-coordinate step bound eta / sqrt(1 - beta2), with bias correction
for b2 in [0.9, 0.99, 0.999]:
    v = 0.0
    for k in range(1, 3000):
        g = rng.standard_normal() * (1000.0 if rng.random() < 0.01 else 1e-3)
        v = b2 * v + (1 - b2) * g * g
        vh = v / (1 - b2**k)
        assert abs(g) / math.sqrt(vh) <= 1 / math.sqrt(1 - b2) + 1e-9
print("OK  Adam(beta1=0) step bound |dw| <= eta/sqrt(1-beta2)")
print("all checks passed")
