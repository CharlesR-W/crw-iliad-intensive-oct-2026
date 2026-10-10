#!/usr/bin/env python3
"""Numerical checks for WB2 3.D (benign overfitting: a junk tail acts as ridge), wb2_new_benign.py.

Run: python checks/wb2_new_benign_check.py   (numpy only; under a minute).
Model: x = (x_s, x_t), x_s ~ N(0, I_k), x_t ~ N(0, nu I_m), y = x_s.beta_s + noise(sigma^2),
n samples, min-norm interpolant b = X^T (X X^T)^{-1} y. Tail total variance c = m nu.
Each check prints the claim, the predicted value, the simulated value, and PASS/FAIL.
"""
import numpy as np

rng = np.random.default_rng(1)
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def draw(n, k, m, nu, beta_s, sigma):
    Xs = rng.standard_normal((n, k))
    Xt = np.sqrt(nu) * rng.standard_normal((n, m))
    y = Xs @ beta_s + sigma * rng.standard_normal(n)
    return Xs, Xt, y


def minnorm(Xs, Xt, y):
    """Signal coefficients, tail test cost nu*||b_t||^2 (via u), and the training residual."""
    G = Xs @ Xs.T + Xt @ Xt.T
    u = np.linalg.solve(G, y)
    bs = Xs.T @ u
    Ktu = Xt @ (Xt.T @ u)          # X_t X_t^T u  (avoid forming the m x m or n x n product twice)
    return bs, u, Ktu


def ridge(Xs, y, a):
    k = Xs.shape[1]
    return np.linalg.solve(Xs.T @ Xs + a * np.eye(k), Xs.T @ y)


k, n, sigma = 5, 100, 1.0
beta_s = np.ones(k)                 # r^2 = ||beta_s||^2 = 5
r2 = beta_s @ beta_s

# ---------- 3.D.1.1: push-through; exact ridge when X_t X_t^T = c I ----------
Xs = rng.standard_normal((n, k)); y = rng.standard_normal(n); c = 7.0
lhs = Xs.T @ np.linalg.solve(Xs @ Xs.T + c * np.eye(n), y)
rhs = ridge(Xs, y, c)
check("1.1 X_s^T(X_sX_s^T+cI)^{-1}y = (X_s^TX_s+cI)^{-1}X_s^T y", np.allclose(lhs, rhs, atol=1e-10),
      f"max diff {np.abs(lhs-rhs).max():.2e}")
# Build an exact tail with X_t X_t^T = c I: X_t = sqrt(c) * (n x m matrix with orthonormal rows)
m = 300
Q, _ = np.linalg.qr(rng.standard_normal((m, n)))
Xt = np.sqrt(c) * Q.T
bs, u, Ktu = minnorm(Xs, Xt, y)
check("1.1 min-norm signal block = ridge(n*lambda=c) when X_tX_t^T=cI exactly", np.allclose(bs, rhs, atol=1e-10),
      f"max diff {np.abs(bs-rhs).max():.2e}")
res = y - Xs @ rhs
check("3.D.2.3 X_t b_t = ridge residual (exact tail)", np.allclose(Ktu, res, atol=1e-10),
      f"max diff {np.abs(Ktu-res).max():.2e}")
nu = c / m
tail = nu * (u @ Ktu)
check("3.D.2.3 nu||b_t||^2 = ||ridge residual||^2/m (exact tail)", np.isclose(tail, res @ res / m, rtol=1e-10),
      f"{tail:.6g} vs {res @ res / m:.6g}")
check("interpolation: train residual of min-norm is 0", np.allclose(Xs @ bs + Ktu, y, atol=1e-10),
      f"max |y - Xb| {np.abs(Xs @ bs + Ktu - y).max():.2e}")

# ---------- 3.D.1.2: concentration of W/m, W = sum_j z_j z_j^T ----------
for m in [400, 2000, 20000]:
    T = 300 if m <= 2000 else 40
    dev_v, opn = [], []
    v = np.zeros(n); v[0] = 1.0
    for _ in range(T):
        Z = rng.standard_normal((n, m))
        D = Z @ Z.T / m - np.eye(n)
        dev_v.append(np.sum((D @ v) ** 2))
        ev = np.linalg.eigvalsh(D + np.eye(n))
        opn.append((ev.min(), ev.max()))
    pred = (n + 1) / m
    sim = np.mean(dev_v)
    check(f"1.2 E||(W/m-I)v||^2 = (n+1)/m  [m={m}]", abs(sim - pred) < 0.08 * pred, f"pred {pred:.4g}, sim {sim:.4g}")
    lo, hi = np.mean(opn, axis=0)
    q = np.sqrt(n / m)
    check(f"1.2 eigenvalues of W/m near [(1-sqrt(n/m))^2,(1+sqrt(n/m))^2]  [m={m}]",
          abs(lo - (1 - q) ** 2) < 0.06 and abs(hi - (1 + q) ** 2) < 0.06,
          f"edges sim [{lo:.3f},{hi:.3f}] vs MP [{(1-q)**2:.3f},{(1+q)**2:.3f}]")

# ---------- 3.D.2.4: pure-noise tail (k=0): E nu y^T (X_tX_t^T)^{-1} y = sigma^2 n/(m-n-1), any nu ----------
for m in [110, 150, 400, 2000]:
    for nu in [1.0, 1e-3]:
        T = 2000 if m <= 400 else 400
        vals = []
        for _ in range(T):
            Xt = np.sqrt(nu) * rng.standard_normal((n, m))
            yy = sigma * rng.standard_normal(n)
            vals.append(nu * yy @ np.linalg.solve(Xt @ Xt.T, yy))
        pred = sigma ** 2 * n / (m - n - 1)
        sim = np.mean(vals)
        se = np.std(vals) / np.sqrt(T)
        check(f"2.4 pure-noise tail cost = sigma^2 n/(m-n-1)  [m={m}, nu={nu:g}]",
              abs(sim - pred) < 4 * se + 0.02 * pred, f"pred {pred:.4g}, sim {sim:.4g} +- {se:.2g}")

# ---------- main simulation: k=5, n=100, m up to 20000, nu = c/m ----------
print("\n  c      m     risk(min-norm)  risk(ridge)  signal-part   tail-part   RSS_ridge/m  bound ||y||^2/m  "
      "||bs-ridge||/||ridge-beta||")
for c in [1.0, 10.0, 50.0]:
    a = c
    approx_ridge = (a ** 2 * r2 + sigma ** 2 * n * k) / (n + a) ** 2
    for m in [200, 1000, 5000, 20000]:
        nu = c / m
        T = 200 if m <= 5000 else 60
        R_mn, R_rg, R_sig, R_tail, R_rss, R_bnd, rel = [], [], [], [], [], [], []
        for _ in range(T):
            Xs, Xt, y = draw(n, k, m, nu, beta_s, sigma)
            bs, u, Ktu = minnorm(Xs, Xt, y)
            br = ridge(Xs, y, a)
            sig = np.sum((bs - beta_s) ** 2)
            tail = nu * (u @ Ktu)
            R_sig.append(sig); R_tail.append(tail); R_mn.append(sig + tail)
            R_rg.append(np.sum((br - beta_s) ** 2))
            res = y - Xs @ br
            R_rss.append(res @ res / m)
            R_bnd.append(y @ y / m)
            rel.append(np.linalg.norm(bs - br) / np.linalg.norm(br - beta_s))
        print(f"{c:5.0f} {m:6d}   {np.mean(R_mn):10.4f}   {np.mean(R_rg):10.4f}   {np.mean(R_sig):10.4f}  "
              f"{np.mean(R_tail):10.5f}   {np.mean(R_rss):10.5f}   {np.mean(R_bnd):10.5f}       {np.mean(rel):.3f}")
        if m == 20000:
            # signal block ~ ridge; the remaining gap of the total is the tail cost ~ sigma^2 (n-k)/m -> 0
            check(f"main: min-norm signal part ~ ridge risk at n*lambda=c  [c={c:g}, m=20000]",
                  abs(np.mean(R_sig) - np.mean(R_rg)) < 0.05 * np.mean(R_rg),
                  f"{np.mean(R_sig):.4f} vs {np.mean(R_rg):.4f}")
            check(f"main: min-norm total risk - ridge risk ~ tail cost  [c={c:g}, m=20000]",
                  abs(np.mean(R_mn) - np.mean(R_rg) - np.mean(R_tail)) < 0.05 * np.mean(R_rg),
                  f"{np.mean(R_mn) - np.mean(R_rg):.4f} vs {np.mean(R_tail):.4f}")
            check(f"main: tail cost <= bound ||y||^2/m * (1+d)^2/(1-d)^4, d=sqrt(n/m)  [c={c:g}]",
                  np.mean(R_tail) <= np.mean(R_bnd) * (1 + np.sqrt(n/m))**2 / (1 - np.sqrt(n/m))**4,
                  f"{np.mean(R_tail):.5f} <= {np.mean(R_bnd) * (1 + np.sqrt(n/m))**2 / (1 - np.sqrt(n/m))**4:.5f}")
            check(f"main: tail cost ~ RSS_ridge/m  [c={c:g}, m=20000]",
                  abs(np.mean(R_tail) - np.mean(R_rss)) < 0.1 * np.mean(R_rss),
                  f"{np.mean(R_tail):.5f} vs {np.mean(R_rss):.5f}")
            if c <= 10:  # only claimed for c << n
                check(f"main: tail cost ~ sigma^2 (n-k)/m (small c)  [c={c:g}, m=20000]",
                  abs(np.mean(R_tail) - sigma**2 * (n - k) / m) < 0.15 * sigma**2 * (n - k) / m,
                  f"{np.mean(R_tail):.5f} vs {sigma**2*(n-k)/m:.5f}")
            check(f"main: ridge risk ~ (c^2 r^2 + sigma^2 n k)/(n+c)^2  [c={c:g}]",
                  abs(np.mean(R_rg) - approx_ridge) < 0.2 * approx_ridge,
                  f"sim {np.mean(R_rg):.4f} vs approx {approx_ridge:.4f}")
    print()

# ---------- contrast: isotropic min-norm near p = n (variance peak) ----------
# Same n, all p features N(0,1), signal in first k: risk explodes near p = n.
for p in [110, 130, 400]:
    T = 2000
    vals = []
    for _ in range(T):
        X = rng.standard_normal((n, p))
        beta = np.zeros(p); beta[:k] = beta_s
        y = X @ beta + sigma * rng.standard_normal(n)
        b = X.T @ np.linalg.solve(X @ X.T, y)
        vals.append(np.sum((b - beta) ** 2))
    pred = r2 * (1 - n / p) + sigma ** 2 * n / (p - n - 1)
    check(f"contrast isotropic min-norm risk = r^2(1-n/p) + sigma^2 n/(p-n-1)  [p={p}]",
          abs(np.mean(vals) - pred) < 0.1 * pred, f"pred {pred:.3f}, sim {np.mean(vals):.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
