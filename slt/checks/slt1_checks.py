#!/usr/bin/env python3
"""Numerical checks of the closed forms used in NB1's solutions (run: python checks/slt1_checks.py)."""
import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng(0)

# Exercise 1.3.4: exact free energy of the conjugate model against Laplace: difference O(1/n).
tau2 = 4.0
for n in [20, 200, 2000]:
    x = rng.normal(1.0, 1.0, n); xbar = x.mean()
    nL = 0.5 * n * np.log(2 * np.pi) + 0.5 * np.sum((x - xbar) ** 2)
    F_exact = nL + 0.5 * np.log(n) + 0.5 * np.log(tau2 + 1 / n) + xbar**2 / (2 * (tau2 + 1 / n))
    F_lap = nL + 0.5 * np.log(n) + 0.5 * np.log(tau2) + xbar**2 / (2 * tau2)
    w = np.linspace(-10, 10, 200001)
    logf = -0.5 * w**2 / tau2 - 0.5 * np.log(2 * np.pi * tau2) - 0.5 * n * np.log(2 * np.pi) - 0.5 * ((x[None] - w[:, None]) ** 2).sum(1)
    F_quad = -(logsumexp(logf) + np.log(w[1] - w[0]))
    assert abs(F_quad - F_exact) < 1e-6, (n, F_quad, F_exact)
    assert abs(F_lap - F_exact) * n < 2.0, (n, (F_lap - F_exact) * n)
print("1.3.4 exact = quadrature; Laplace error * n bounded: ok")

# Exercise 1.4.4: the three generalization errors against the table, by Monte Carlo over the sample mean.
sigma2 = 2.0
for n in [10, 100]:
    xbar = rng.normal(0, np.sqrt(sigma2 / n), 200000)
    v = 1 / n
    S = 0.5 * np.log(2 * np.pi * np.e * sigma2); Lmin = 0.5 * (sigma2 - 1 - np.log(sigma2))
    bayes = 0.5 * np.log(2 * np.pi * (1 + v)) + (sigma2 + xbar**2) / (2 * (1 + v)) - S - Lmin
    gibbs = 0.5 * np.log(2 * np.pi) + 0.5 * (sigma2 + xbar**2 + v) - S - Lmin
    mle = 0.5 * np.log(2 * np.pi) + 0.5 * (sigma2 + xbar**2) - S - Lmin
    assert abs(n * bayes.mean() - 0.5 * n * np.log(1 + 1 / n)) < 0.02
    assert abs(n * gibbs.mean() - (1 + sigma2) / 2) < 0.02
    assert abs(n * mle.mean() - sigma2 / 2) < 0.02
print("1.4.4 Bayes 1/2n, Gibbs (1+sigma^2)/2n, MLE sigma^2/2n: ok")

# Exercise 1.4.3, Bayes row: direct Monte Carlo of KL(q || p*) for the Gaussian location model with q = N(0, sigma2),
# averaged over samples, equals Lmin + 1/(2n) + O(1/n^2) independent of sigma2.
for sigma2 in [0.5, 2.0]:
    n = 50
    Lmin = 0.5 * (sigma2 - 1 - np.log(sigma2))
    xbar = rng.normal(0, np.sqrt(sigma2 / n), 100000)
    v = 1 / n
    G = 0.5 * np.log(2 * np.pi * (1 + v)) + (sigma2 + xbar**2) / (2 * (1 + v)) - 0.5 * np.log(2 * np.pi * np.e * sigma2)
    assert abs((G.mean() - Lmin) * n - 0.5) < 0.02, (sigma2, (G.mean() - Lmin) * n)
print("1.4.3 Bayes predictive excess d/2n independent of sigma^2: ok")
