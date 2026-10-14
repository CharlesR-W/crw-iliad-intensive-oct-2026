#!/usr/bin/env python3
"""Numerical checks for the formulas stated in slt4_applications (NB4).

1. Susceptibility (§2): for p_h(w) ∝ φ(w) exp(-nβ L_h(w)) with L_h = (1-h) L + h L',
   χ = (1/nβ) d<O>_h/dh |_{h=0} = -Cov_β(O, ΔL), ΔL = L' - L.
2. Bayesian influence function (§3): for p(w) ∝ φ(w) exp(-Σ_i β_i ℓ_i(w)),
   d<O>/dβ_i |_{β=1} = -Cov(ℓ_i, O).
   Loss kernel K_ij = Cov(ℓ_i, ℓ_j) is PSD, and K_ij = -BIF(z_i, O=ℓ_j).
3. MDL (§4): regular case λ = d/2 recovers the BIC term (d/2) log n; checked on
   K(w) = |w|^2 in d = 3 by the volume law V(t) ∝ t^{d/2}.

All on a 1-D or 2-D grid, by finite differences against the covariance formula.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1-D grid model
w = np.linspace(-3, 3, 20001)
dw = w[1] - w[0]
phi = np.exp(-w**2 / 2)                       # prior (unnormalised)
L = 0.5 * (w**2 - 1) ** 2 + 0.1 * w           # some loss
Lp = 0.3 * (w - 0.5) ** 2 + 0.2 * np.sin(3 * w)  # perturbed-data loss L'
obs = np.tanh(2 * w) + 0.3 * w**2             # observable ϕ
n, beta = 50.0, 0.7


def expect(f, logweight):
    lw = logweight - logweight.max()
    p = np.exp(lw) * phi
    return (f * p).sum() / p.sum()


def cov(f, g, logweight):
    return expect(f * g, logweight) - expect(f, logweight) * expect(g, logweight)


def mean_obs(h):
    Lh = (1 - h) * L + h * Lp
    return expect(obs, -n * beta * Lh)


eps = 1e-5
chi_fd = (mean_obs(eps) - mean_obs(-eps)) / (2 * eps) / (n * beta)
chi_cov = -cov(obs, Lp - L, -n * beta * L)
print(f"susceptibility: finite diff {chi_fd:.8f}  -Cov {chi_cov:.8f}")
assert abs(chi_fd - chi_cov) < 1e-6 * max(1, abs(chi_cov))

# ---------------------------------------------------------------- BIF on a small regression
xs = rng.normal(size=8)
ys = 1.3 * xs + 0.3 * rng.normal(size=8)
W1, W2 = np.meshgrid(np.linspace(-1, 3, 601), np.linspace(-2, 2, 601), indexing="ij")
prior2 = np.exp(-(W1**2 + W2**2) / 8)
ell = np.stack([0.5 * (y - W1 * x - W2 * x**3) ** 2 for x, y in zip(xs, ys)])  # ℓ_i(w), w=(a,b)


def expect2(f, betas):
    lw = -np.tensordot(betas, ell, axes=1)
    lw -= lw.max()
    p = np.exp(lw) * prior2
    return (f * p).sum() / p.sum()


ones = np.ones(len(xs))
obs2 = W1 * W2 + np.cos(W1)
i = 3
e = np.zeros(len(xs)); e[i] = 1e-5
bif_fd = (expect2(obs2, ones + e) - expect2(obs2, ones - e)) / 2e-5
bif_cov = -(expect2(ell[i] * obs2, ones) - expect2(ell[i], ones) * expect2(obs2, ones))
print(f"BIF: finite diff {bif_fd:.8f}  -Cov {bif_cov:.8f}")
assert abs(bif_fd - bif_cov) < 1e-6 * max(1, abs(bif_cov))

means = np.array([expect2(l, ones) for l in ell])
Kmat = np.array([[expect2(a * b, ones) for b in ell] for a in ell]) - np.outer(means, means)
eig = np.linalg.eigvalsh(Kmat)
print(f"loss kernel min eigenvalue {eig.min():.3e} (PSD up to rounding)")
assert eig.min() > -1e-10 * eig.max()
j = 5
e = np.zeros(len(xs)); e[i] = 1e-5
bif_ij = (expect2(ell[j], ones + e) - expect2(ell[j], ones - e)) / 2e-5
print(f"K_ij {Kmat[i, j]:.8f}  -BIF(z_i, ℓ_j) {-bif_ij:.8f}")
assert abs(Kmat[i, j] + bif_ij) < 1e-6 * max(1, abs(Kmat[i, j]))

# ---------------------------------------------------------------- regular volume law
d = 3
pts = rng.uniform(-1, 1, size=(2_000_000, d))
Kv = (pts**2).sum(1)
ts = np.array([1e-3, 4e-3, 1.6e-2, 6.4e-2])
V = np.array([(Kv < t).mean() for t in ts])
slope = np.polyfit(np.log(ts), np.log(V), 1)[0]
print(f"volume exponent for K=|w|^2 in d={d}: {slope:.3f} (d/2 = {d/2})")
assert abs(slope - d / 2) < 0.05
print("all checks passed")
