"""Checks for the derivation deck, lectures/L0_deriving_the_notes.html.

Run:  python da/checks/L0_checks.py     (numpy, scipy; about 2 s)
Each check prints PASS/FAIL; the script exits nonzero on any failure.
"""
import itertools
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

FAILS = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(1)

# §1.3 Shapley on the croissant: butter, yeast get 1/2; flour, sugar 0.
players = ["flour", "butter", "yeast", "sugar"]
v = lambda S: float("butter" in S or "yeast" in S)
n = 4
for p in players:
    others = [q for q in players if q != p]
    phi = sum((v(set(S) | {p}) - v(set(S))) / (n * math.comb(n - 1, len(S)))
              for k in range(n) for S in itertools.combinations(others, k))
    perm = np.mean([v(set(o[:o.index(p) + 1])) - v(set(o[:o.index(p)]))
                    for o in itertools.permutations(players)])
    check(f"Shapley {p}", abs(phi - perm) < 1e-12 and abs(phi - (0.5 if p in ("butter", "yeast") else 0)) < 1e-12)

# Example 2.1: weighted mean derivative (z_i - zbar)/n, and -H^{-1} g_i gives the same.
z = rng.normal(size=7)
nz = len(z)
wstar = lambda b: (b @ z) / b.sum()
e = 1e-6
for i in range(nz):
    b = np.ones(nz); b[i] += e
    fd = (wstar(b) - z.mean()) / e
    check(f"mean influence i={i}", abs(fd - (z[i] - z.mean()) / nz) < 1e-5)

# Ex. 2.2: OLS reweighting closed form, Taylor coefficient, LOO 1/(1-h).
N, p = 15, 3
X = rng.normal(size=(N, p)); y = rng.normal(size=N)
A = X.T @ X
ws = np.linalg.solve(A, X.T @ y)
j = 4
r = y[j] - X[j] @ ws
h = X[j] @ np.linalg.solve(A, X[j])
pj = np.linalg.solve(A, X[j])
for bj in [1.0, 0.7, 0.3, 0.0]:
    W = np.ones(N); W[j] = bj
    wb = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * y))
    closed = -(1 - bj) * pj * r / (1 - (1 - bj) * h)
    check(f"OLS reweight closed form beta_j={bj}", np.allclose(wb - ws, closed))
W = np.ones(N); W[j] = 1 + e
wb = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * y))
check("OLS IF = A^{-1} x_j r_j", np.allclose((wb - ws) / e, pj * r, atol=1e-5))
check("leverage in (0,1)", 0 < h < 1, f"h={h:.3f}")

# Ex. 2.3: the a.b toy.
zs = rng.normal(loc=2.0, size=9); zb = zs.mean(); nn = len(zs)
for a in [np.sqrt(zb), zb, 0.5]:
    bb = zb / a
    vv = np.array([bb, a])
    Hn = sum(np.array([[bb**2, 2 * a * bb - zi], [2 * a * bb - zi, a**2]]) for zi in zs)
    check(f"ab Hessian = n v v^T (a={a:.2f})", np.allclose(Hn, nn * np.outer(vv, vv)))
    check("null space (a,-b)", np.allclose(Hn @ np.array([a, -bb]), 0))
    zi = zs[0]
    gi = -(zi - zb) * vv
    Iplus = -vv @ np.linalg.pinv(Hn) @ gi
    check("pinv IF = (z_i - zbar)/n", abs(Iplus - (zi - zb) / nn) < 1e-10)
    for lam in [1e-3, 1.0, 10.0]:
        Ml = np.linalg.inv(Hn + lam * np.eye(2))
        check(f"damped IF phi=ab lam={lam}", abs(-vv @ Ml @ gi - (zi - zb) * (vv @ vv) / (nn * (vv @ vv) + lam)) < 1e-10)
        ga2 = np.array([2 * a, 0.0])
        check(f"damped IF phi=a^2 lam={lam}", abs(-ga2 @ Ml @ gi - 2 * a * bb * (zi - zb) / (nn * (vv @ vv) + lam)) < 1e-10)
loo = (zs.sum() - zs[0]) / (nn - 1) - zb
check("ab LOO = -(z_i - zbar)/(n-1)", abs(loo + (zs[0] - zb) / (nn - 1)) < 1e-12)

# §2.2 (2): G != H at a critical point. f = w^2, target -1, L = (w^2+1)^2/2 at w = 0.
Lw = lambda w: 0.5 * (w**2 + 1) ** 2
w0, hh = 0.0, 1e-4
gL = (Lw(w0 + hh) - Lw(w0 - hh)) / (2 * hh)
HL = (Lw(w0 + hh) - 2 * Lw(w0) + Lw(w0 - hh)) / hh**2
GL = (2 * w0) ** 2 * 1.0   # J^T l'' J with f = w^2, l = (f + 1)^2 / 2
check("GN counterexample: grad 0, H = 2, G = 0", abs(gL) < 1e-8 and abs(HL - 2) < 1e-4 and GL == 0)

# Example 3.2: Bayesian linear regression BIF = (A + sigma^2/tau^2 I)^{-1} x_j e_j.
sig, tau = 0.7, 1.5
def post(Wt):
    V = np.linalg.inv((X.T * Wt) @ X / sig**2 + np.eye(p) / tau**2)
    return V @ (X.T @ (Wt * y)) / sig**2, V
mu, V = post(np.ones(N))
W = np.ones(N); W[j] += e
mu_e, _ = post(W)
ej = y[j] - X[j] @ mu
check("BIF Bayesian OLS = d mu / d beta_j", np.allclose((mu_e - mu) / e, V @ X[j] * ej / sig**2, atol=1e-5))
check("... = (A + lambda I)^{-1} x_j e_j", np.allclose(V @ X[j] * ej / sig**2, np.linalg.solve(A + sig**2 / tau**2 * np.eye(p), X[j] * ej)))

# Ex. 3.2(d): Isserlis (2,2) term; Monte Carlo against 1/2 tr(Hphi S Hi S).
d = 3
Ssym = rng.normal(size=(d, d)); Sig = Ssym @ Ssym.T / d + 0.3 * np.eye(d)
Hp = rng.normal(size=(d, d)); Hp = Hp + Hp.T
Hi = rng.normal(size=(d, d)); Hi = Hi + Hi.T
D = rng.multivariate_normal(np.zeros(d), Sig, size=2_000_000)
qa = 0.5 * np.einsum("na,ab,nb->n", D, Hp, D); qb = 0.5 * np.einsum("na,ab,nb->n", D, Hi, D)
mc = np.cov(qa, qb)[0, 1]; ex = 0.5 * np.trace(Hp @ Sig @ Hi @ Sig)
check("Isserlis (2,2) = 1/2 tr(Hphi S Hi S)", abs(mc - ex) < 0.02 * (1 + abs(ex)), f"MC {mc:.4f} vs {ex:.4f}")

# Ex. 3.3(b): inf of a nonquadratic L - eps L_k matches the expansion to O(eps^3).
from scipy.optimize import minimize
Hq = np.array([[3.0, 0.5], [0.5, 1.0]]); gk = np.array([0.8, -0.4])
Lf = lambda w: 0.5 * w @ Hq @ w + 0.3 * w[0] ** 3 + 0.2 * w[1] ** 4
Lk = lambda w: gk @ w + 0.5 * w[0] ** 2
for eps in [1e-2, 5e-3]:
    val = minimize(lambda w: Lf(w) - eps * Lk(w), np.zeros(2), tol=1e-14).fun
    pred = -eps * Lk(np.zeros(2)) - 0.5 * eps**2 * gk @ np.linalg.solve(Hq, gk)
    check(f"inf(L - eps L_k) expansion eps={eps}", abs(val - pred) < 50 * eps**3, f"err {abs(val - pred):.1e}")
check("L=w^4, L_k=w: inf = -3(eps/4)^{4/3}",
      abs(minimize(lambda w: w[0] ** 4 - 0.01 * w[0], [0.1], tol=1e-14).fun + 3 * (0.01 / 4) ** (4 / 3)) < 1e-10)

# Ex. 3.3(e): Gaussian KL with covariance change.
k = 2
for eps in [1e-3]:
    Wt = np.ones(N); Wt[k] -= eps
    mu_eps, V_eps = post(Wt)
    W2 = np.ones(N); W2[k] += 1e-7
    BIF = (post(W2)[0] - mu) / 1e-7
    rr = rng.normal(size=p)
    m0, m1 = mu + eps * rr, mu_eps
    Vi = np.linalg.inv(V_eps)
    kl = 0.5 * (np.trace(Vi @ V) - p + (m1 - m0) @ Vi @ (m1 - m0) + np.log(np.linalg.det(V_eps) / np.linalg.det(V)))
    M = np.outer(X[k], X[k]) @ V / sig**2
    pred = 0.5 * eps**2 * (rr + BIF) @ np.linalg.inv(V) @ (rr + BIF) + 0.25 * eps**2 * np.trace(M @ M)
    check("3.3(e) KL formula", abs(kl - pred) < 1e-3 * eps**2 + 1e-12, f"{kl:.3e} vs {pred:.3e}")
    check("3.3(e) mu_eps = mu - eps BIF", np.allclose((mu_eps - mu) / eps, -BIF, atol=1e-2))
    check("3.3(e) V^{-1}(-BIF) = grad L_k(mu)", np.allclose(np.linalg.inv(V) @ (-BIF), -X[k] * (y[k] - X[k] @ mu) / sig**2, atol=1e-5))

# Ex. 4.1: unrolling formula against finite differences, minibatch SGD on logistic regression.
Nl, dl, T, b = 12, 3, 40, 4
Xl = rng.normal(size=(Nl, dl)); yl = (rng.random(Nl) < 0.5).astype(float)
batches = [rng.choice(Nl, b, replace=False) for _ in range(T)]
etas = 0.5 * np.ones(T)
sigm = lambda u: 1 / (1 + np.exp(-u))
gradi = lambda w, i: (sigm(Xl[i] @ w) - yl[i]) * Xl[i]
hessi = lambda w, i: sigm(Xl[i] @ w) * (1 - sigm(Xl[i] @ w)) * np.outer(Xl[i], Xl[i])
def run(beta):
    w = np.zeros(dl); traj = [w]
    for t in range(T):
        w = w - etas[t] / b * sum(beta[i] * gradi(w, i) for i in batches[t]); traj.append(w)
    return traj
traj = run(np.ones(Nl))
i0 = 5
Js = [np.eye(dl) - etas[t] / b * sum(hessi(traj[t], i) for i in batches[t]) for t in range(T)]
tot = np.zeros(dl)
for t in range(T):
    if i0 in batches[t]:
        P = np.eye(dl)
        for s in range(t + 1, T):
            P = Js[s] @ P
        tot += -etas[t] / b * P @ gradi(traj[t], i0)
bp = np.ones(Nl); bp[i0] += 1e-6; bm = np.ones(Nl); bm[i0] -= 1e-6
fd = (run(bp)[-1] - run(bm)[-1]) / 2e-6
check("unrolling formula = finite differences", np.allclose(tot, fd, atol=1e-7), f"max err {np.abs(tot - fd).max():.1e}")

# SOURCE geometric series and filter limits.
Hb = np.diag([0.01, 0.5, 3.0]); eta, K = 0.1, 200
Mx = np.eye(3) - eta * Hb
lhs = eta * sum(np.linalg.matrix_power(Mx, kk) for kk in range(K))
check("geometric series (I - M^K)(I-M)^{-1}", np.allclose(lhs, (np.eye(3) - np.linalg.matrix_power(Mx, K)) @ np.linalg.inv(Hb)))
Fr = lambda s: (1 - np.exp(-eta * K * s)) / s
check("F_r(0+) = eta K, F_r(large) ~ 1/s", abs(Fr(1e-9) - eta * K) < 1e-4 and abs(Fr(10.0) * 10 - 1) < 1e-6)
check("F_r at crossover = (1 - 1/e) eta K", abs(Fr(1 / (eta * K)) - (1 - np.exp(-1)) * eta * K) < 1e-12)

# Ex. 4.2(b): converged window formula with inherited response.
Hs = A; gk2 = rng.normal(size=p); alpha = 1 / np.linalg.eigvalsh(Hs).max(); Kw = 50
r0 = rng.normal(size=p); rcur = r0.copy()
for _ in range(Kw):
    rcur = (np.eye(p) - alpha * Hs) @ rcur + alpha * gk2
MK = np.linalg.matrix_power(np.eye(p) - alpha * Hs, Kw)
check("window: r_T = M^K r + (I - M^K) H^{-1} g", np.allclose(rcur, MK @ r0 + (np.eye(p) - MK) @ np.linalg.solve(Hs, gk2)))

# §4.4: SVD derivative, A_12 = -1/8 for S = diag(3,1), E = e1 e2^T.
S = np.diag([3.0, 1.0]); E = np.array([[0.0, 1.0], [0.0, 0.0]])
U0, s0, Vt0 = np.linalg.svd(S); U1, s1, Vt1 = np.linalg.svd(S + 1e-7 * E)
sgn = np.sign(np.diag(U0.T @ U1)); U1 = U1 * sgn; Vt1 = (Vt1.T * sgn).T
Afd = U0.T @ (U1 - U0) / 1e-7
pred = (1.0 * E[0, 1] + 3.0 * E[1, 0]) / (1.0**2 - 3.0**2)
check("SVD derivative A_12 = -1/8", abs(Afd[0, 1] - pred) < 1e-5 and abs(pred + 1 / 8) < 1e-12, f"fd {Afd[0, 1]:.5f}")

# §4.4: logistic mode solution solves tau G' = 2 G (s - G).
sk, G0 = 2.0, 1e-3
Gt = lambda t: sk * np.exp(2 * sk * t) / (np.exp(2 * sk * t) - 1 + sk / G0)
ts = np.linspace(0, 5, 11)
dG = (Gt(ts + 1e-6) - Gt(ts - 1e-6)) / 2e-6
check("mode solution satisfies logistic ODE", np.allclose(dG, 2 * Gt(ts) * (sk - Gt(ts)), rtol=1e-5))

# §4.4: dog-on-cat plateau value and zero endpoint (same set-up as assets/L0/make_figures.py).
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "lectures" / "assets" / "L0"))
import make_figures as mf  # noqa: E402
import matplotlib
matplotlib.use("Agg")
mf.save = lambda fig, name: None
res = mf.dog_cat()
check("dog/cat plateau within 5% of formula", abs(res["mid"] - res["plateau"]) < 0.05 * res["plateau"], f"{res['mid']:.4f} vs {res['plateau']:.4f}")
check("dog/cat endpoint ~ 0", abs(res["final"]) < 1e-4)
check("dog/cat helpful during mode-1 rise", res["mn"] < 0)

print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
