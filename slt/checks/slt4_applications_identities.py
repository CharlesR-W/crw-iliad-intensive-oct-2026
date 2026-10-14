#!/usr/bin/env python3
"""Check the one identity stated in slt4_applications (§4.2, reused in §4.3).

For E_h under p_h(w) ∝ φ(w) exp(-β (n L_n(w) + h Δ(w))):
    d/dh E_h[O] |_{h=0} = -β Cov_β(O, Δ).
Special cases: Δ = n(L' - L) (susceptibility, Baker et al.: χ = (1/nβ) dE/dh = -Cov(O, L'-L));
Δ = ℓ_i, one sample's loss (Bayesian influence function, Kreer et al.: -Cov(O, ℓ_i) at β = 1).
Adapted from _local_prior_2026-10-12/checks/slt4_applications_identities.py. Run from slt/.
"""
import numpy as np

rng = np.random.default_rng(0)
w = np.linspace(-3, 3, 20001)
phi = np.exp(-w**2 / 2)
L = 0.5 * (w**2 - 1) ** 2 + 0.1 * w           # training loss L_n
Lp = 0.3 * (w - 0.5) ** 2 + 0.2 * np.sin(3 * w)  # loss on shifted data L'
O = np.tanh(2 * w) + 0.3 * w**2
n, beta = 50.0, 0.7


def E(f, energy):
    lw = -beta * energy
    p = np.exp(lw - lw.max()) * phi
    return (f * p).sum() / p.sum()


def cov(f, g, energy):
    return E(f * g, energy) - E(f, energy) * E(g, energy)


def check(Delta, name, scale=1.0):
    eps = 1e-5
    fd = (E(O, n * L + eps * Delta) - E(O, n * L - eps * Delta)) / (2 * eps)
    rhs = -beta * cov(O, Delta, n * L)
    print(f"{name}: finite difference {fd / scale:.8f}   formula {rhs / scale:.8f}")
    assert abs(fd - rhs) < 1e-6 * max(1, abs(rhs))


check(np.sin(w) + 0.2 * w**3, "general Delta")
check(n * (Lp - L), "susceptibility chi = (1/n beta) dE/dh", scale=n * beta)

# BIF: per-sample losses of a small 1-D regression, beta = 1 so -Cov(O, l_i) exactly.
beta = 1.0
xs = rng.normal(size=8)
ys = 1.3 * xs + 0.3 * rng.normal(size=8)
ell = np.stack([0.5 * (y - w * x) ** 2 for x, y in zip(xs, ys)])
L = ell.sum(0) / n
check(ell[3], "BIF (beta = 1): dE/d(weight of sample 3)")
print("all checks passed")
