#!/usr/bin/env python3
"""Checks of the claims made in NB1's figure captions (run from slt/: python checks/slt1_bayes_statmech_figures.py)."""
import numpy as np

rng = np.random.default_rng(0)

# dictionary: the deep narrow well U holds less posterior mass than the shallow wide well V.
w = np.linspace(-3, 3, 6001)
g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
E = -np.logaddexp(-0.5 * ((w - 1.3) / 0.16) ** 2, np.log(0.45) - 0.5 * ((w + 1.1) / 0.5) ** 2)
post = g(w, 0, 1.4) * np.exp(-E)
mid = np.abs(w - 0.3) < 0.6
cut = w[mid][np.argmin(post[mid])]
ZU, ZV = np.trapezoid(post[w >= cut], w[w >= cut]), np.trapezoid(post[w <= cut], w[w <= cut])
assert E[w > cut].min() < E[w < cut].min() and ZU < ZV, (ZU, ZV)
print(f"dictionary: U deeper, Z(U)={ZU:.3f} < Z(V)={ZV:.3f}: ok")

# covariances: model N(mu, e^{2s}) on data Exp(1)-1. w0 = (0, 0), H = diag(1, 2),
# C = [[1, skew], [skew, kurt-1]] = [[1, 2], [2, 8]], sandwich H^-1 C H^-1 = [[1, 1], [1, 2]].
H = np.diag([1.0, 2.0]); C = np.array([[1.0, 2.0], [2.0, 8.0]])
Hi = np.linalg.inv(H)
assert np.allclose(Hi @ C @ Hi, [[1, 1], [1, 2]])
n, reps = 400, 20000
x = rng.exponential(1, (reps, n)) - 1
pts = np.sqrt(n) * np.c_[x.mean(1), 0.5 * np.log(x.var(1))]
emp = np.cov(pts.T)
assert np.allclose(emp, [[1, 1], [1, 2]], atol=0.12), emp
print("covariances: sqrt(n)(w_hat - w0) covariance", emp.round(2).tolist(),
      "vs posterior H^-1 = diag(1, 0.5): s-variance 4x, sd 2x larger: ok")
xw = rng.normal(0, 1, (reps, n))
emp_w = np.cov((np.sqrt(n) * np.c_[xw.mean(1), 0.5 * np.log(xw.var(1))]).T)
assert np.allclose(emp_w, Hi, atol=0.06), emp_w
print("covariances: well specified matches H^-1:", emp_w.round(2).tolist(), "ok")

# occam: V_post = int exp(-n H (w - w_hat)^2 / 2) dw = sqrt(2 pi / (n H)); 4n halves it.
for nn, Hn in [(50, 2.0), (200, 2.0)]:
    u = np.linspace(-5, 5, 200001)
    Vp = np.trapezoid(np.exp(-0.5 * nn * Hn * u**2), u)
    assert abs(Vp - np.sqrt(2 * np.pi / (nn * Hn))) < 1e-9
print("occam: V_post = sqrt(2 pi / n H), halves when n -> 4n: ok")

# predictors: flat prior, model N(x; w, 1): posterior N(xbar, 1/n), predictive N(xbar, 1 + 1/n),
# and the average of the six quantile members approximates it.
nn, xbar = 4, 0.4
q = xbar + np.sqrt(1 / nn) * np.array([-1.383, -0.674, -0.210, 0.210, 0.674, 1.383])
xs = np.linspace(-6, 6, 4001)
mix = np.mean([g(xs, m, 1.0) for m in q], axis=0)
exact = g(xs, xbar, np.sqrt(1 + 1 / nn))
assert np.max(np.abs(mix - exact)) < 0.01, np.max(np.abs(mix - exact))
print("predictors: mean of quantile members ~ N(xbar, 1+1/n): ok")
