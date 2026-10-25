#!/usr/bin/env python3
"""Numerical/symbolic checks for L1_attribution.html, including speaker notes.

Run from the repository root: python da/checks/L1_checks.py
CPU, a few seconds; numpy, scipy, sympy and torch required. No files are written.
Summed losses; influence is the upweighting derivative; deletion predicts -I.
The slow, basin-sensitive MLP figure reproduction remains in make_figures.py;
this script instead checks PBRF on a small non-stationary nonlinear network.
"""
import itertools
import math
import sys
import time

import numpy as np
import sympy as sp
import torch
from scipy.special import expit
from scipy.stats import spearmanr

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
START = time.perf_counter()
FAIL = []
COUNT = 0


def check(name, ok, detail=""):
    global COUNT
    COUNT += 1
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


# finite-coalition attribution and the weighted-mean warm-up.
players = ("butter", "yeast", "flour", "sugar")


def value(S):
    return int(bool(set(S) & {"butter", "yeast"}))


shapley = []
for player in players:
    others = [p for p in players if p != player]
    shapley.append(sum(
        (value((*S, player)) - value(S)) / math.comb(3, k)
        for k in range(4) for S in itertools.combinations(others, k)
    ) / 4)
check("croissant Shapley = (1/2, 1/2, 0, 0)", np.allclose(shapley, [.5, .5, 0, 0]))
check("croissant leave-one-out credits nobody",
      all(value(players) - value(set(players) - {p}) == 0 for p in players))
zs = sp.symbols("z0:4", real=True)
bs = sp.symbols("b0:4", positive=True)
mu = sum(b * z for b, z in zip(bs, zs)) / sum(bs)
check("weighted mean derivative = (z_i - zbar)/n",
      all(sp.simplify(sp.diff(mu, bs[i]).subs(dict.fromkeys(bs, 1))
                      - (zs[i] - sum(zs) / 4) / 4) == 0 for i in range(4)))
mm = sp.symbols("mu", real=True)
loss_mean = sum((z - mm)**2 / 2 for z in zs)
check("mean IFT: H=n and -H^-1 g_i=(z_i-zbar)/n",
      sp.diff(loss_mean, mm, 2) == 4 and
      sp.simplify(-sp.diff((zs[0] - mm)**2 / 2, mm).subs(mm, sum(zs) / 4) / 4
                  - (zs[0] - sum(zs) / 4) / 4) == 0)

# EXACT DA1 section 3 / make_figures.py RNG and data order.
rng = np.random.default_rng(0)
n = 15
x = np.r_[rng.normal(0, 1, n - 1), 7.0]
y = 1.0 + 0.6 * x + 0.4 * rng.normal(size=n)
y[-1] -= 3.0
X = np.c_[np.ones(n), x]
A = X.T @ X
w = np.linalg.solve(A, X.T @ y)
residual = y - X @ w
hat = X @ np.linalg.solve(A, X.T)
h = np.diag(hat)
upweight = (np.linalg.solve(A, X.T) * residual).T
pred = -upweight
exact = np.array([np.linalg.lstsq(np.delete(X, j, 0), np.delete(y, j), rcond=None)[0] - w
                  for j in range(n)])
check("OLS hat matrix is an orthogonal projection", np.allclose(hat, hat.T) and np.allclose(hat @ hat, hat))
check("OLS leverage 0 <= h <= 1; sum h = p", np.all(h >= 0) and np.all(h <= 1) and np.isclose(h.sum(), 2))
check("OLS exact deletion = -A^-1 x_j r_j / (1-h_j)",
      np.allclose(exact, pred / (1 - h[:, None]), atol=1e-12))
ok_response = True
for j in range(n):
    for b in (0.0, 0.4, 1.001, 1.7):
        beta = np.ones(n); beta[j] = b
        refit = np.linalg.solve((X.T * beta) @ X, X.T @ (beta * y))
        ok_response &= np.allclose(refit - w, -(1 - b) * upweight[j] / (1 - (1 - b) * h[j]), atol=1e-12)
check("OLS full response curve agrees with re-solved weighted fits", ok_response)
check("1/(1-h) geometric series for all observed h<1",
      np.allclose(sum(h**k for k in range(300)), 1 / (1 - h), atol=1e-12))
hh = sp.symbols("h", real=True)
check("geometric-series remainder = h^K/(1-h)",
      sp.simplify(1 / (1 - hh) - sum(hh**k for k in range(8)) - hh**8 / (1 - hh)) == 0)
X_edge = np.array([[1., 0.], [2., 0.], [0., 1.]])
h_edge = np.diag(X_edge @ np.linalg.solve(X_edge.T @ X_edge, X_edge.T))
check("h=1 permits full rank before deletion but loses rank afterwards",
      h_edge[-1] == 1 and np.linalg.matrix_rank(X_edge) == 2 and np.linalg.matrix_rank(X_edge[:-1]) == 1)
check("leverage = uncentered Mahalanobis squared / n",
      np.allclose(h, np.einsum("ij,ji->i", X, np.linalg.solve(A / n, X.T)) / n))
check("intercept leverage = 1/n + centered Mahalanobis squared / n",
      np.allclose(h, 1 / n + (x - x.mean())**2 / (n * x.var())))
jA, jB = int(np.argmin(np.abs(x[:-1]))), n - 1
ratio = 1 / (1 - h[jB])
slopes = np.array([w[1], w[1] + exact[jB, 1], w[1] + pred[jB, 1]])
check("quoted slopes: full 0.256, LOO 0.674, tangent 0.330",
      np.array_equal(np.round(slopes, 3), [.256, .674, .330]), str(slopes.round(9)))
check("quoted leverages A=0.07, B=0.83; magnification 5.7 (5.72 in notes)",
      round(h[jA], 2) == .07 and round(h[jB], 2) == .83 and round(ratio, 1) == 5.7 and round(ratio, 2) == 5.72,
      f"h_A={h[jA]:.9f}, h_B={h[jB]:.9f}, ratio={ratio:.9f}")

# the global minimum switches wells, while the local branch persists.
def wells(tilt):
    roots = np.roots([4., 0., -4., tilt])
    minima = roots[np.isreal(roots) & (12 * roots.real**2 - 4 > 0)].real
    return minima[np.argmin((minima**2 - 1)**2 + tilt * minima)]


check("double-well: beta_3 1 -> .8 switches the global minimum",
      wells(.10 - .11) > 0 and wells(.10 - .8 * .11) < 0)

# singular a*b model, symbolic identities and both lambda limits.
a, b, zbar, zi, nn, lam = sp.symbols("a b zbar zi n lambda", positive=True)
v = sp.Matrix([b, a])
# The total squared loss differs from n/2 (ab-zbar)^2 only by a constant.
H_ab = sp.hessian(nn * (a * b - zbar)**2 / 2, (a, b))
at_min = {b: zbar / a}
check("a*b Hessian at a minimum = n v v^T",
      sp.simplify((H_ab - nn * v * v.T).subs(at_min)) == sp.zeros(2))
gi_ab = sp.Matrix([sp.diff((a * b - zi)**2 / 2, u) for u in (a, b)])
check("a*b individual gradient = -(z_i-zbar) v",
      sp.simplify((gi_ab + (zi - zbar) * v).subs(at_min)) == sp.zeros(2, 1))
check("a*b flat direction (a,-b) is orthogonal to g_i",
      (v.T * sp.Matrix([a, -b]))[0] == 0)
vv = (v.T * v)[0]
check("a*b damped solve on v = v/(n ||v||^2 + lambda)",
      sp.simplify((nn * v * v.T + lam * sp.eye(2)) * v / (nn * vv + lam) - v) == sp.zeros(2, 1))
for name, gp, expected in (
    ("ab", v, (zi - zbar) / nn),
    ("a^2", sp.Matrix([2 * a, 0]), 2 * zbar * (zi - zbar) / (nn * vv)),
):
    damped = (zi - zbar) * (gp.T * v)[0] / (nn * vv + lam)
    check(f"a*b lambda->0 limit for phi={name}",
          sp.simplify((sp.limit(damped, lam, 0) - expected).subs(at_min)) == 0)
z_ab = 2 + np.random.default_rng(1).normal(size=10)
zb, delta = z_ab.mean(), z_ab.max() - z_ab.mean()
for a0 in (.6, 1., np.sqrt(zb), 2.5):
    b0 = zb / a0
    vn = np.array([b0, a0])
    Hn = len(z_ab) * np.outer(vn, vn)
    gin = -delta * vn
    gps = np.array([vn, [2 * a0, 0.]])
    expected = delta * (gps @ vn) / (len(z_ab) * (vn @ vn) + .3)
    check(f"a*b explicit damped formula at a={a0:.3f}",
          np.allclose(-gps @ np.linalg.solve(Hn + .3 * np.eye(2), gin), expected))
    # Gradient flow conserves a^2-b^2. Its nearby endpoint is determined by
    # this invariant and ab=weighted mean; differentiate that exact endpoint.
    invariant = a0*a0 - b0*b0
    def endpoint(beta):
        mean = (z_ab.sum() + (beta - 1) * z_ab.max()) / (len(z_ab) + beta - 1)
        a_squared = (invariant + np.sqrt(invariant**2 + 4 * mean**2)) / 2
        return np.array([mean, a_squared])
    eps = 1e-5
    fd = (endpoint(1 + eps) - endpoint(1 - eps)) / (2 * eps)
    limit = delta * (gps @ vn) / (len(z_ab) * (vn @ vn))
    check(f"a*b both zero-damping limits match warm-flow endpoint derivative at a={a0:.3f}",
          np.allclose(fd, limit, atol=1e-9))

# nonlinear network, GN decomposition and genuine PBRF refits.
torch.manual_seed(17)
xt = torch.tensor([-.9, -.2, .4, 1.1])
yt = torch.tensor([.3, -.6, .8, -.1])
ws = torch.randn(7) * .4  # two tanh units, scalar output; deliberately untrained


def net(theta, inputs=xt):
    return torch.tanh(inputs[:, None] * theta[:2] + theta[2:4]) @ theta[4:6] + theta[6]


def individual(theta):
    return .5 * (net(theta) - yt)**2


grad = torch.func.grad
hessian = torch.func.hessian
J = torch.func.jacrev(net)(ws)
G = J.T @ J
H = hessian(lambda t: individual(t).sum())(ws)
output_hessians = torch.func.jacrev(torch.func.jacrev(net))(ws)
R = torch.einsum("n,npq->pq", net(ws) - yt, output_hessians)
check("tiny network: autodiff Hessian = GN + residual term", torch.allclose(H, G + R, atol=1e-12))
check("tiny network: G is PSD and residual term is nonzero", torch.linalg.eigvalsh(G).min() > -1e-12 and R.norm() > .1)
ww = sp.symbols("w", real=True)
check("f=w^2, target -1: strict minimum w=0 has H=2, G=0",
      sp.diff((ww**2 + 1)**2 / 2, ww).subs(ww, 0) == 0 and
      sp.diff((ww**2 + 1)**2 / 2, ww, 2).subs(ww, 0) == 2 and
      (sp.diff(ww**2, ww)**2).subs(ww, 0) == 0)
fs = net(ws).detach()
lam_p = .7
eye = torch.eye(len(ws))


def Q(theta, beta):
    # Squared-error Bregman divergence equals .5 ||f-fs||^2.
    return .5 * ((net(theta) - fs)**2).sum() + ((beta - 1) * individual(theta)).sum() + .5 * lam_p * ((theta - ws)**2).sum()


ones = torch.ones(len(xt))
train_norm = grad(lambda t: individual(t).sum())(ws).norm().item()
check("PBRF checkpoint is non-stationary for the training loss", train_norm > .1, f"||grad L||={train_norm:.6f}")
check("Q(w_s;1)=0, grad Q(w_s;1)=0",
      Q(ws, ones) == 0 and grad(Q)(ws, ones).norm() < 1e-12)
# Global optimality at beta=1 follows from the displayed sum of nonnegative
# squares, and lambda>0 makes equality possible only at ws. Also probe points.
probes = ws + torch.randn(20, len(ws))
check("Q(w;1) >= lambda/2 ||w-w_s||^2, uniquely minimized at w_s",
      all(Q(t, ones) >= .5 * lam_p * ((t - ws)**2).sum() for t in probes))
check("PBRF Hessian at checkpoint = G + lambda I",
      torch.allclose(hessian(Q)(ws, ones), G + lam_p * eye, atol=1e-12))
gi = torch.func.jacrev(individual)(ws)
mixed = torch.func.jacrev(grad(Q), argnums=1)(ws, ones)
check("PBRF mixed derivative d_beta grad_w Q = per-example gradients", torch.allclose(mixed, gi.T))


def solve_Q(beta):
    theta = ws.clone()
    for _ in range(15):
        gradient = grad(Q)(theta, beta)
        if gradient.norm() < 1e-13:
            break
        theta -= torch.linalg.solve(hessian(Q)(theta, beta), gradient)
    return theta, grad(Q)(theta, beta).norm().item(), torch.linalg.eigvalsh(hessian(Q)(theta, beta)).min().item()


def phi_t(theta):
    return .5 * (net(theta, torch.tensor([.65]))[0] - .25)**2


gp = grad(phi_t)(ws)
response = -torch.linalg.solve(G + lam_p * eye, gi.T)
eps = 2e-5
fd_param, fd_phi, solve_norms, eigmins = [], [], [], []
for i in range(len(xt)):
    shift = torch.zeros(len(xt)); shift[i] = eps
    wp, np_, ep = solve_Q(ones + shift)
    wm, nm, em = solve_Q(ones - shift)
    fd_param.append((wp - wm) / (2 * eps))
    fd_phi.append((phi_t(wp) - phi_t(wm)) / (2 * eps))
    solve_norms += [np_, nm]; eigmins += [ep, em]
check("perturbed PBRF refits are stationary strict local minima", max(solve_norms) < 1e-11 and min(eigmins) > 0)
check("PBRF parameter derivative agrees with independently re-solved finite differences",
      torch.allclose(torch.stack(fd_param).T, response, atol=2e-8, rtol=2e-7))
check("d(phi o PBRF)/d beta_i = -g_phi^T (G+lambda I)^-1 g_i",
      torch.allclose(torch.stack(fd_phi), gp @ response, atol=2e-8, rtol=2e-7))

# differentiate actual GD trajectories, not just the recurrence.
Hq = np.array([[2., .3], [.3, .7]])
gq = np.array([.4, -.9])
eta, K = .15, 23


def unroll(beta, matrix=Hq, forcing=gq):
    theta = np.array([.3, -.1])
    for _ in range(K):
        # Quadratic L=.5 w^T H w + (beta-1) g^T w, constant H and g.
        theta -= eta * (matrix @ theta + (beta - 1) * forcing)
    return theta


filter_response = -(np.eye(2) - np.linalg.matrix_power(np.eye(2) - eta * Hq, K)) @ np.linalg.solve(Hq, gq)
eps = 1e-5
check("unrolled quadratic GD finite differences = -[I-(I-eta H)^K] H^-1 g",
      np.allclose((unroll(1 + eps) - unroll(1 - eps)) / (2 * eps), filter_response, atol=1e-10))
Hnull = np.diag([0., 2.])
null_fd = (unroll(1 + eps, Hnull) - unroll(1 - eps, Hnull)) / (2 * eps)
check("unrolled GD null-space response = -eta K g", np.isclose(null_fd[0], -eta * K * gq[0]))
s, T = sp.symbols("s T", positive=True)
check("continuous-time GD filter limit F(0)=T", sp.limit((1 - sp.exp(-T * s)) / s, s, 0) == T)
check("stable discrete GD filter approaches 1/s", np.isclose((1 - (1 - eta * 2)**1000) / 2, .5))
sigma2, tau2 = sp.symbols("sigma2 tau2", positive=True)
check("Bayesian linear regression in X^T X units: F(0)=tau^2/sigma_noise^2",
      (1 / (s + sigma2 / tau2)).subs(s, 0) == tau2 / sigma2)

# reproduce DA1 section 5 INCLUDING RNG draws,
# fixed ridge, epsilon, subset sizes, per-query metrics and centered slopes.
rng = np.random.default_rng(1)
n, d, kappa, nq = 80, 10, .01, 10
scale = np.logspace(0, -1.5, d)
X = rng.normal(size=(n, d)) * scale
w_true = 4 * rng.normal(size=d) / np.sqrt(scale)
y = (rng.random(n) < expit(X @ w_true)).astype(float)
Xq = rng.normal(size=(nq, d)) * scale
yq = (rng.random(nq) < expit(Xq @ w_true)).astype(float)
fit_residuals = []


def fit(beta, w0=None, lam=0., center=None):
    w = np.zeros(d) if w0 is None else w0.copy()
    for _ in range(100):
        p = expit(X @ w)
        g = X.T @ (beta * (p - y)) + kappa * w
        H = (X.T * (beta * p * (1 - p))) @ X + kappa * np.eye(d)
        if lam:
            g += lam * (w - center)
            H += lam * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-14 * (1 + np.abs(w).max()):
            break
    final_g = X.T @ (beta * (expit(X @ w) - y)) + kappa * w
    if lam:
        final_g += lam * (w - center)
    fit_residuals.append(np.linalg.norm(final_g))
    return w


def phi(w):
    logits = Xq @ w
    return np.logaddexp(0, logits) - yq * logits


w_star = fit(np.ones(n))
p = expit(X @ w_star)
grads = X * (p - y)[:, None]
H = (X.T * (p * (1 - p))) @ X + kappa * np.eye(d)
gphi = Xq * (expit(Xq @ w_star) - yq)[:, None]
ev = np.linalg.eigvalsh(H)
lam = np.median(ev)
scores = {
    "exact IF": -gphi @ np.linalg.solve(H, grads.T),
    "damped IF": -gphi @ np.linalg.solve(H + lam * np.eye(d), grads.T),
    "grad dot": -gphi @ grads.T,
}
check("logistic IFT problem is strictly convex (fixed positive ridge)", ev.min() >= kappa)
eps = 1e-3
fd, fd_prox, loo = (np.zeros((nq, n)) for _ in range(3))
param_fd, prox_param_fd = [], []
for i in range(n):
    e = np.zeros(n); e[i] = eps
    wp, wm = fit(1 + e, w_star), fit(1 - e, w_star)
    wpp, wmp = fit(1 + e, w_star, lam, w_star), fit(1 - e, w_star, lam, w_star)
    fd[:, i] = (phi(wp) - phi(wm)) / (2 * eps)
    fd_prox[:, i] = (phi(wpp) - phi(wmp)) / (2 * eps)
    param_fd.append((wp - wm) / (2 * eps))
    prox_param_fd.append((wpp - wmp) / (2 * eps))
    beta = np.ones(n); beta[i] = 0
    loo[:, i] = phi(fit(beta, w_star)) - phi(w_star)
check("IFT parameter derivative -H^-1 g_i matches re-solved minima",
      np.allclose(np.array(param_fd).T, -np.linalg.solve(H, grads.T), atol=2e-6, rtol=2e-5))
check("damping is exact derivative of stationary-point proximal objective",
      np.allclose(np.array(prox_param_fd).T, -np.linalg.solve(H + lam * np.eye(d), grads.T), atol=2e-6, rtol=2e-5))
S = np.zeros((100, n), bool)
for subset in S:
    subset[rng.choice(n, n // 2, replace=False)] = True
out = np.array([phi(fit(subset.astype(float), w_star)) for subset in S])
check("all 501 logistic fits converged", len(fit_residuals) == 501 and max(fit_residuals) < 1e-10,
      f"max gradient norm {max(fit_residuals):.2e}")


def relerr(I, target):
    return np.mean(np.linalg.norm(I - target, axis=1) / np.linalg.norm(target, axis=1))


def slope(xv, yv):
    xc = xv - xv.mean()
    return xc @ (yv - yv.mean()) / (xc @ xc)


expected = {"exact IF": (.96, .83), "damped IF": (.82, .83), "grad dot": (.36, .56)}
subset_slopes = []
for name, I in scores.items():
    rho = np.mean([spearmanr(-I[q], loo[q]).statistic for q in range(nq)])
    predictions = S.astype(float) @ I.T
    lds = np.mean([spearmanr(predictions[:, q], out[:, q]).statistic for q in range(nq)])
    calibration = np.mean([slope(predictions[:, q], out[:, q]) for q in range(nq)])
    subset_slopes.append(calibration)
    check(f"quoted logistic {name}: LOO / LDS", (round(rho, 2), round(lds, 2)) == expected[name],
          f"{rho:.9f} / {lds:.9f}; subset calibration slope {calibration:.4f}")
err = relerr(scores["exact IF"], fd)
err_prox = relerr(scores["damped IF"], fd_prox)
check("quoted own-target relative errors both round to 3e-7",
      f"{err:.0e}" == "3e-07" and f"{err_prox:.0e}" == "3e-07", f"{err:.9e}, {err_prox:.9e}")
wrong_target = relerr(scores["damped IF"], fd)
check("quoted damped IF error vs plain derivative = 48%", round(100 * wrong_target) == 48, f"{100 * wrong_target:.5f}%")
condition = ev[-1] / ev[0]
check("quoted Hessian eigenvalue ratio about 360", abs(condition - 360) < 5, f"{condition:.6f}")
check("quoted half-subset calibration range about 12 to 35",
      abs(min(subset_slopes) - 12) < 1 and abs(max(subset_slopes) - 35) < 1)
check("LDS sign: subset sum = deletion prediction plus a constant",
      np.allclose(S @ scores["exact IF"].T, -((~S) @ scores["exact IF"].T) + scores["exact IF"].sum(1)))
# Koh--Liang epsilon weights mean loss: beta_i=1+n*epsilon.
check("Koh--Liang normalization: n*I with epsilon=-1/n predicts -I",
      np.allclose((-1 / n) * (n * scores["exact IF"]), -scores["exact IF"]))

print(f"SUMMARY: {COUNT - len(FAIL)}/{COUNT} checks passed; {len(FAIL)} failed.")
print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
print(f"Runtime: {time.perf_counter() - START:.2f}s (CPU; excludes imports)")
sys.exit(1 if FAIL else 0)
