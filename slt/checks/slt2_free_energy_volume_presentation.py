#!/usr/bin/env python3
"""Checks for claims added in the 2026-10-14 presentation pass of NB2 (figures, captions, prose).
Run from slt/:  python checks/slt2_free_energy_volume_presentation.py
"""
import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng(1)
N = 4_000_000
a, b = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)

# sublevel2d caption: as t drops by 4, V drops by 4 (bowl), 4^{3/4} (a^2+b^4), and by less (ab)^2.
ts = [1 / 4, 1 / 16, 1 / 64]
for name, K, ratio in [("bowl", a**2 + b**2, 4.0), ("a2+b4", a**2 + b**4, 4**0.75), ("cross", (a * b) ** 2, None)]:
    V = [np.mean(K < t) for t in ts]
    r = [V[0] / V[1], V[1] / V[2]]
    if ratio is not None:
        assert all(abs(x / ratio - 1) < 0.03 for x in r), (name, r)
    else:
        assert all(x < 2.0 for x in r), (name, r)
    print(f"{name}: V = {np.round(V, 4)}, ratios {np.round(r, 2)}")
# closed forms used in the logV figure (uniform prior on [-1,1]^2, t<1)
t = 1e-3
I = np.trapezoid(np.sqrt(1 - np.linspace(-1, 1, 200001) ** 4), np.linspace(-1, 1, 200001))
assert abs(np.mean(a**2 + b**2 < t) / (np.pi * t / 4) - 1) < 0.03
assert abs(np.mean(a**2 + b**4 < t) / (0.5 * I * t**0.75) - 1) < 0.03
assert abs(np.mean((a * b) ** 2 < t) / (0.5 * np.sqrt(t) * -np.log(t) + np.sqrt(t)) - 1) < 0.01
# local slope of log V for (ab)^2 approaches 1/2 from below, slowly
for lt in (-2, -5, -10):
    t = 10.0**lt
    V = lambda t: 0.5 * np.sqrt(t) * -np.log(t) + np.sqrt(t)
    s = (np.log(V(t * 1.01)) - np.log(V(t))) / np.log(1.01)
    assert 0.3 < s < 0.5, s
print("logV closed forms and (ab)^2 local slope < 1/2: ok")

# two_basins figure: posterior mass in w>0 at n=10 and n=300 for delta=0.03 (uniform prior on the grid)
delta = 0.03
w = np.linspace(-2.0, 2.5, 200001)
x = np.clip(w + 0.5, 0, 1)
K = (w + 1) ** 2 * (w - 1) ** 4 + delta * x**3 * (10 - 15 * x + 6 * x**2)
# the step is exactly constant on both basins, so K - delta = (w+1)^2 (w-1)^4 near w = 1 (lambda_V = 1/4 exactly)
assert np.all(x[w >= 0.5] == 1) and np.all(x[w <= -0.5] == 0)
for n, lo, hi in [(10, 0.7, 0.9), (300, 0.0, 0.005)]:
    lp = -n * K
    m = np.exp(logsumexp(lp[w > 0]) - logsumexp(lp))
    assert lo < m < hi, (n, m)
    print(f"two basins, n={n}: mass in V = {m:.3f}")
# local lambdas of the two basins: K ~ 16 (w+1)^2 at -1, K - delta ~ 4 (w-1)^4 at +1
assert abs(((-1 + 1e-3 + 1) ** 2 * (-1 + 1e-3 - 1) ** 4) / (16e-6) - 1) < 0.01
assert abs(((1 + 1e-2 + 1) ** 2 * (1e-2) ** 4) / (4e-8) - 1) < 0.02

# thermodynamic paragraph: log-odds 0.2 n - 18 log n pass from -5 to 5 over about 10/Delta L = 50 samples
f = lambda n: 0.2 * n - 18 * np.log(n)
ns = np.linspace(300, 1000, 700001)
lo, hi = ns[np.argmin(abs(f(ns) + 5))], ns[np.argmin(abs(f(ns) - 5))]
assert 50 < hi - lo < 60, hi - lo
print(f"Exercise 2.3.2 window: log-odds -5..5 over n in [{lo:.0f}, {hi:.0f}], width {hi - lo:.0f}")

# Exercise 2.3.4 crossovers (leading terms): n/log n = 18.75, 100, 35 -> n ~ 83, 650, 180
from scipy.optimize import brentq
for target, expect in [(18.75, 83), (100, 650), (35, 180)]:
    nx = brentq(lambda n: n / np.log(n) - target, 3, 1e6)
    assert abs(nx / expect - 1) < 0.02, (target, nx)
print("Exercise 2.3.4 crossovers 83, 647, 182: ok")
# Exercise 2.2.3: for K = w^2 + v^4 on [-1,1]^2, V(t) = I t^{3/4} / 2 exactly for t <= 1 (region inside the box)
t = 0.9
assert abs(np.mean(a**2 + b**4 < t) / (0.5 * I * t**0.75) - 1) < 0.01
print("2.2.3 V exact for t<=1: ok")
# Prior positivity matters for lambda <= d/2: K = w^2, prior 3w^2/2 on [-1,1] gives V(t) = t^{3/2}
u = np.linspace(-1, 1, 2_000_001); t = 1e-2
assert abs(np.trapezoid(np.where(u**2 < t, 1.5 * u**2, 0), u) / t**1.5 - 1) < 1e-3
print("prior 3w^2/2: V = t^(3/2): ok")

# Exercise 2.3.1 sign: with Delta L > 0 and Delta lambda < 0 (opposite signs) there is no crossover
f2 = lambda n: 0.2 * n + 3 * np.log(n)
assert np.all(f2(np.linspace(1.01, 1e5, 10000)) > 0)
print("Exercise 2.3.1 opposite signs -> no crossover: ok")
