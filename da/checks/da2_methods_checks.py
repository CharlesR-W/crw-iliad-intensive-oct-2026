#!/usr/bin/env python3
"""Numerical and symbolic checks for every closed-form result stated in da2_methods.

Run: python da/checks/da2_methods_checks.py   (CPU, ~20 s). Each check prints PASS/FAIL
with the quantity compared. Conventions are DA1's: summed loss L(w;beta)=sum_i beta_i L_i,
upweighting derivative at beta=1, deletion beta_i: 1 -> 0; mean-loss SGD shows 1/n.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


# ======================================================================= stop 1: EK-FAC
def stop1():
    m, d, N = 3, 4, 2000
    a = rng.standard_normal((N, d)) @ rng.standard_normal((d, d))
    g = rng.standard_normal((N, m)) @ rng.standard_normal((m, m))
    g = g * (1 + 0.8 * np.tanh(a[:, :1]))          # correlate g with a (breaks independence)
    # per-example gradient of W (m x d), row-major flatten: vec(g a^T) = kron(g, a)
    D = np.einsum("ni,nj->nij", g, a).reshape(N, -1)
    check("vec(g a^T) == kron(g, a) (row-major)", np.allclose(D[0], np.kron(g[0], a[0])))
    G = D.T @ D / N
    S, A = g.T @ g / N, a.T @ a / N
    lS, QS = np.linalg.eigh(S)
    lA, QA = np.linalg.eigh(A)
    Q = np.kron(QS, QA)
    lam = 0.3
    v = rng.standard_normal(m * d)
    # damped inverse of S (x) A by two small eigendecompositions, damping on joint eigenvalues
    V = v.reshape(m, d)
    Vt = QS.T @ V @ QA
    Vt = Vt / (np.outer(lS, lA) + lam)
    fast = (QS @ Vt @ QA.T).reshape(-1)
    direct = np.linalg.solve(np.kron(S, A) + lam * np.eye(m * d), v)
    check("K-FAC damped IVP via two eigendecompositions", np.allclose(fast, direct))
    factored = np.linalg.solve(np.kron(S + np.sqrt(lam) * np.eye(m), A + np.sqrt(lam) * np.eye(d)), v)
    check("factor-wise damping is a different operator", not np.allclose(factored, direct),
          f"(rel diff {np.linalg.norm(factored - direct) / np.linalg.norm(direct):.2f})")
    # EK-FAC eigenvalues: second moments of gradients in the Kronecker eigenbasis
    proj = np.einsum("ia,nij,jb->nab", QS, np.einsum("ni,nj->nij", g, a), QA)
    Lam = (proj ** 2).mean(0)
    check("EK-FAC Lambda = diag(Q^T G Q)", np.allclose(Lam.reshape(-1), np.diag(Q.T @ G @ Q)))
    best = np.linalg.norm(G - Q @ np.diag(Lam.reshape(-1)) @ Q.T)
    worse = [np.linalg.norm(G - Q @ np.diag(Lam.reshape(-1) * (1 + 0.05 * rng.standard_normal(m * d))) @ Q.T)
             for _ in range(50)]
    check("EK-FAC diagonal is Frobenius-optimal in that basis", best <= min(worse))
    check("||G - EKFAC|| <= ||G - KFAC||", best <= np.linalg.norm(G - np.kron(S, A)))
    # under independence the refit reproduces the K-FAC eigenvalues
    a2 = rng.standard_normal((200000, d))
    g2 = rng.standard_normal((200000, m)) * np.array([1.0, 2.0, 0.5])
    proj2 = (g2 @ QS) ** 2
    Lam2 = np.einsum("ni,nj->ij", proj2, (a2 @ QA) ** 2) / len(a2)
    kfac2 = np.outer(((g2 @ QS) ** 2).mean(0), ((a2 @ QA) ** 2).mean(0))
    check("independent a, g: EK-FAC eigenvalues == K-FAC eigenvalues (MC)",
          np.allclose(Lam2, kfac2, rtol=0.03))


# ======================================================================= stop 2: unrolling
def stop2():
    d, n = 5, 7
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    Hbar = X.T @ X / n + 0.05 * np.eye(d)            # mean Hessian (ridge 0.05)
    eta = 0.5 / np.linalg.eigvalsh(Hbar).max()
    wstar = np.linalg.solve(Hbar, X.T @ y / n)
    k = 2
    gk = -(y[k] - X[k] @ wstar) * X[k]

    def gd(eps, w0, K):
        beta = np.ones(n)
        beta[k] -= eps
        w = w0.copy()
        for _ in range(K):
            w = w - eta * (X.T @ (beta * (X @ w - y)) / n + 0.05 * w)
        return w

    K = 40
    e = 1e-6
    r_fd = (gd(e, wstar, K) - gd(-e, wstar, K)) / (2 * e)
    M = np.linalg.matrix_power(np.eye(d) - eta * Hbar, K)
    r_formula = np.linalg.solve(Hbar, (np.eye(d) - M) @ gk) / n
    check("downweight response from w*: (1/n)[I-(I-eta H)^K] H^-1 g_k (sign +, factor 1/n)",
          np.allclose(r_fd, r_formula, atol=1e-8))
    # inherited response: start from a w0 != w*, run T-K then K steps
    w0 = rng.standard_normal(d)
    T = 400
    wTK = gd(0, w0, T - K)
    r_TK = (gd(e, w0, T - K) - gd(-e, w0, T - K)) / (2 * e)
    r_T = (gd(e, w0, T) - gd(-e, w0, T)) / (2 * e)
    g_window = -(y[k] - X[k] @ wTK) * X[k]   # nearly frozen: w is close to w* after T-K steps
    pred = M @ r_TK + np.linalg.solve(Hbar, (np.eye(d) - M) @ g_window) / n
    check("window formula with inherited term (w near w*)", np.allclose(pred, r_T, atol=1e-6),
          f"(err {np.linalg.norm(pred - r_T):.1e}, |r_T| {np.linalg.norm(r_T):.2e})")
    # deletion is eps = 1: linear prediction H_sum^{-1} g_k; IF limit
    r_inf = np.linalg.solve(n * Hbar, gk)
    check("K->inf: r = (1/n) Hbar^-1 g_k = H_sum^-1 g_k = -(upweighting IF)",
          np.allclose(np.linalg.solve(Hbar, gk) / n, r_inf))
    # null-space drift for PSD H with g_k having a null component
    Hs = np.diag([1.0, 0.5, 0.0])
    g = np.array([0.3, -0.2, 0.1])
    r = np.zeros(3)
    for _ in range(1000):
        r = (np.eye(3) - 0.1 * Hs) @ r + 0.1 * g        # n = 1 here
    check("null-space component grows linearly: P0 r_K = K eta P0 g", np.isclose(r[2], 1000 * 0.1 * 0.1))
    # SOURCE filter numbers
    F = lambda s, eK: -np.expm1(-eK * s) / s
    eK = 7.0
    check("SOURCE filter at s=1/(eta K): 0.632 eta K; ridge 1/(s+1/(eta K)) gives 0.5 eta K",
          np.isclose(F(1 / eK, eK) / eK, 1 - np.exp(-1)) and np.isclose(1 / (2 / eK) / eK, 0.5))
    check("SOURCE filter F(0) = eta K", np.isclose(F(1e-12, eK), eK, rtol=1e-6))
    check("frozen GD filter -> SOURCE filter for small eta s",
          np.isclose((1 - (1 - 1e-4 * 0.3) ** 70000) / 0.3, F(0.3, 7.0), rtol=1e-3))
    # preconditioned step: Jacobian includes DP[v] g
    c = 0.7
    w = rng.standard_normal(3)
    Wc = rng.standard_normal((3, 3))
    gradf = lambda w: Wc @ w + np.sin(w)
    Hf = lambda w: Wc + np.diag(np.cos(w))
    P = lambda w: np.diag(1 / (c + w ** 2))
    step = lambda w: w - 0.1 * P(w) @ gradf(w)
    Jfd = np.array([(step(w + 1e-6 * ei) - step(w - 1e-6 * ei)) / 2e-6 for ei in np.eye(3)]).T
    Jfrozen = np.eye(3) - 0.1 * P(w) @ Hf(w)
    DPg = np.diag(-2 * w / (c + w ** 2) ** 2 * gradf(w))       # column v -> DP(w)[v] g
    check("preconditioned Jacobian = I - eta(P H + DP[.] g)", np.allclose(Jfd, Jfrozen - 0.1 * DPg, atol=1e-7))
    check("frozen-preconditioner Jacobian differs when g != 0", not np.allclose(Jfd, Jfrozen, atol=1e-4))


# ======================================================================= stop 3: BIF
def stop3():
    # (A) BIF = -Cov(L_i, phi): 1-D quadrature with a non-Gaussian posterior
    w = np.linspace(-6, 6, 200001)
    Ls = [lambda w: 0.5 * (w - 1) ** 2 + 0.1 * w ** 4, lambda w: np.cosh(w - 0.3), lambda w: 0.2 * (w + 1) ** 2]
    phi = lambda w: np.sin(w) + w ** 2

    def Ephi(beta):
        U = sum(b * L(w) for b, L in zip(beta, Ls))
        p = np.exp(-(U - U.min()))
        return (p * phi(w)).sum() / p.sum(), p / p.sum()

    e = 1e-5
    deriv = (Ephi([1, 1 + e, 1])[0] - Ephi([1, 1 - e, 1])[0]) / (2 * e)
    _, p = Ephi([1, 1, 1])
    L1 = Ls[1](w)
    cov = (p * L1 * phi(w)).sum() - (p * L1).sum() * (p * phi(w)).sum()
    check("BIF = d E[phi]/d beta_i = -Cov(L_i, phi)", np.isclose(deriv, -cov, rtol=1e-6))

    # (B) Bayesian linear regression: BIF on phi = w equals damped IF; units
    n, d, sig, tau = 30, 3, 0.5, 1.3
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + sig * rng.standard_normal(n)
    A = X.T @ X
    V = np.linalg.inv(A / sig ** 2 + np.eye(d) / tau ** 2)
    mu = V @ X.T @ y / sig ** 2
    j = 4
    rj = y[j] - X[j] @ mu
    # exact Gaussian moments: Cov(L_j, w) with L_j = (y_j - x_j^T w)^2/(2 sig^2)
    cov_exact = -(rj / sig ** 2) * V @ X[j]
    bif = -cov_exact
    lam = sig ** 2 / tau ** 2
    check("BLR: BIF(z_j, w) = (A + sigma^2/tau^2 I)^-1 x_j r_j", np.allclose(bif, np.linalg.solve(A + lam * np.eye(d), X[j] * rj)))
    Hlik = A / sig ** 2
    gradLj = -X[j] * rj / sig ** 2
    check("BLR in likelihood units: BIF = -(H_lik + tau^-2 I)^-1 grad L_j", np.allclose(bif, -np.linalg.solve(Hlik + np.eye(d) / tau ** 2, gradLj)))
    # check against the definition by differentiating the conjugate posterior mean
    def post_mean(bj):
        wts = np.ones(n)
        wts[j] = bj
        Vb = np.linalg.inv((X.T * wts) @ X / sig ** 2 + np.eye(d) / tau ** 2)
        return Vb @ (X.T * wts) @ y / sig ** 2
    check("BLR: BIF = d(posterior mean)/d beta_j", np.allclose((post_mean(1 + 1e-6) - post_mean(1 - 1e-6)) / 2e-6, bif, atol=1e-7))
    # quadratic phi in BLR: the (2,2) term survives
    Hphi = np.diag([1.0, 2.0, 0.5])
    Hj = np.outer(X[j], X[j]) / sig ** 2
    Z = rng.multivariate_normal(mu, V, size=400000)
    Lj = (y[j] - Z @ X[j]) ** 2 / (2 * sig ** 2)
    ph = 0.5 * np.einsum("ni,ij,nj->n", Z, Hphi, Z)
    mc = -np.cov(Lj, ph)[0, 1]
    pred = -(mu @ Hphi @ V @ (-X[j] * rj / sig ** 2)) - 0.5 * np.trace(Hphi @ V @ Hj @ V)
    check("BLR, quadratic phi: BIF = -(1,1) - (2,2) exactly (MC)", np.isclose(mc, pred, rtol=0.03), f"(mc {mc:.4f}, formula {pred:.4f})")

    # (C) Gaussian-surrogate terms: odd terms vanish, (1,3),(2,2),(3,1) all O(Sigma^2)
    s = sp.symbols("s", positive=True)
    x = sp.symbols("x")
    # 1-D: Delta ~ N(0, s); T1=g x, T2=h x^2/2, T3=k x^3/6
    gp, hp, kp, gi, hi, ki = sp.symbols("g_p h_p k_p g_i h_i k_i")
    mom = lambda p: 0 if p % 2 else sp.factorial2(p - 1) * s ** (p // 2)
    def E(expr):
        poly = sp.Poly(sp.expand(expr), x)
        return sum(c * mom(m[0]) for m, c in zip(poly.monoms(), poly.coeffs()))
    T = lambda g, h, k: [g * x, h * x ** 2 / 2, k * x ** 3 / 6]
    Tp, Ti = T(gp, hp, kp), T(gi, hi, ki)
    cov = lambda X1, X2: sp.simplify(E(X1 * X2) - E(X1) * E(X2))
    check("(1,1) term = g_phi Sigma g_i", sp.simplify(cov(Tp[0], Ti[0]) - gp * gi * s) == 0)
    check("(1,2),(2,1) vanish", cov(Tp[0], Ti[1]) == 0 and cov(Tp[1], Ti[0]) == 0)
    c13, c22, c31 = cov(Tp[0], Ti[2]), cov(Tp[1], Ti[1]), cov(Tp[2], Ti[0])
    check("(1,3) = 1/2 g_phi k_i Sigma^2", sp.simplify(c13 - gp * ki * s ** 2 / 2) == 0)
    check("(2,2) = 1/2 h_phi h_i Sigma^2", sp.simplify(c22 - hp * hi * s ** 2 / 2) == 0)
    check("(3,1) = 1/2 k_phi g_i Sigma^2", sp.simplify(c31 - kp * gi * s ** 2 / 2) == 0)
    # true posterior vs surrogate at next order: L = n U(w), 1-D
    U = lambda w: 0.5 * w ** 2 + 0.3 * w ** 3 + 0.2 * w ** 4
    Li = lambda w: 0.5 * (w - 0.7) ** 2
    ph = lambda w: w
    ww = np.linspace(-3, 3, 400001)
    res = []
    for nn in [200, 400, 800, 1600]:
        lp = -nn * U(ww)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        c = (p * Li(ww) * ph(ww)).sum() - (p * Li(ww)).sum() * (p * ph(ww)).sum()
        lead = (1.0 / nn) * (-0.7) * 1.0       # g_phi H^-1 g_i, H = n, g_i = -0.7, g_phi = 1
        res.append((nn, (c - lead) * nn ** 2))
    coefs = [r[1] for r in res]
    # surrogate prediction of the n^-2 coefficient: (1,3)=0 (L_i quadratic), (2,2)=0 (phi linear),
    # (3,1)=0 (phi linear) -> surrogate says 0; the true posterior's skewness (U''' != 0) adds a term
    check("linear phi, quadratic L_i: surrogate next order is 0 but true posterior is O(n^-2) != 0",
          abs(coefs[-1]) > 0.1 and abs(coefs[-1] - coefs[-2]) < 0.05 * abs(coefs[-1]),
          f"(n^2 x residual: {', '.join(f'{c:.3f}' for c in coefs)})")
    a_, b_, c_ = 0.3, 0.2, 0.7
    coef = -(6 * a_ + c_ * (36 * a_ ** 2 - 12 * b_))
    check("cumulant formula for the n^-2 coefficient: -(6a + c(36a^2-12b)) = -2.388",
          np.isclose(coef, -2.388) and abs(coefs[-1] - coef) < 0.02, f"(formula {coef:.3f}, quadrature n=1600 {coefs[-1]:.3f})")
    # linear phi does not kill (1,3): L_i with a cubic part
    check("linear phi: (1,3) survives when L_i has a third derivative", sp.simplify(c13.subs({ki: 1, gp: 1})) != 0)
    # L = w^4: isolated minimum, H = 0, posterior width n^-1/4
    widths = []
    for nn in [1e2, 1e4, 1e6]:
        lp = -nn * ww ** 4
        p = np.exp(lp - lp.max())
        p /= p.sum()
        widths.append(np.sqrt((p * ww ** 2).sum()))
    check("L = n w^4: posterior sd scales as n^-1/4", np.isclose(widths[0] / widths[1], 10 ** 0.5, rtol=1e-3))

    # (D) transport: completed square needs the partition term
    H = np.array([[2.0, 0.3], [0.3, 0.7]])
    gk = np.array([0.4, -1.1])
    eps = 1e-3
    r = rng.standard_normal(2)
    energy = 0.5 * eps ** 2 * r @ H @ r - eps ** 2 * gk @ r              # (O(eps) terms cancel with partition)
    part = 0.5 * eps ** 2 * gk @ np.linalg.solve(H, gk)
    sq = 0.5 * eps ** 2 * (r - np.linalg.solve(H, gk)) @ H @ (r - np.linalg.solve(H, gk))
    check("F = energy + partition = (eps^2/2)(r - H^-1 g)^T H (r - H^-1 g)", np.isclose(energy + part, sq))
    rstar = np.linalg.solve(H, gk)
    check("without the partition term the 'cost' can be negative", 0.5 * eps ** 2 * rstar @ H @ rstar - eps ** 2 * gk @ rstar < 0)
    # degenerate case: H^+ grad L_k is not optimal even when L_k is constant on the minimum set
    # (Codex round 2): L = sqrt(1+x^2)-1 + y^4/(1+y^2)^1.5, L_k = tanh y, minimum (0,0), H = diag(1,0)
    Lc = lambda x, y_: np.sqrt(1 + x ** 2) - 1 + y_ ** 4 / (1 + y_ ** 2) ** 1.5
    Lkc = lambda x, y_: np.tanh(y_)
    ok = True
    for eps_ in [1e-1, 1e-2, 1e-3]:
        dF = (Lc(0, eps_) - eps_ * Lkc(0, eps_)) - (Lc(0, 0) - eps_ * Lkc(0, 0))   # F(r=(0,1)) - F(r=0)
        ok &= dF < 0 and np.isclose(dF / eps_ ** 2, -1, rtol=0.02)
    check("degenerate counterexample: r=(0,1) beats r=H^+ grad L_k = 0 by ~eps^2", ok)
    # 1-D low-temperature check of the normalized KL for a constant shift
    L = lambda w: 0.5 * 1.5 * w ** 2 + 0.2 * w ** 3 + 0.1 * w ** 4
    Lk = lambda w: np.sin(w) + 0.3 * w
    g1, H1 = 1.0 + 0.3, 1.5                                          # Lk'(0), L''(0)
    eps = 0.05
    xs = np.linspace(-1.5, 1.5, 300001)
    dx = xs[1] - xs[0]

    def normKL(kappa, r):
        logp0 = -kappa * L(xs)
        logp0 -= np.log(np.exp(logp0 - logp0.max()).sum() * dx) + logp0.max()
        logpe = -kappa * (L(xs) - eps * Lk(xs))
        logpe -= np.log(np.exp(logpe - logpe.max()).sum() * dx) + logpe.max()
        p0 = np.exp(logp0)
        logpe_shift = np.interp(xs + eps * r, xs, logpe)
        return (p0 * (logp0 - logpe_shift)).sum() * dx / kappa
    kap = 4000.0
    rs = np.linspace(0, 2 * g1 / H1, 21)
    vals = [normKL(kap, r) for r in rs]
    check("1-D: normalized KL minimized near r = H^-1 g_k (downweighting shift +)",
          abs(rs[int(np.argmin(vals))] - g1 / H1) < 0.15 * g1 / H1,
          f"(argmin {rs[int(np.argmin(vals))]:.3f}, H^-1 g = {g1 / H1:.3f})")
    # BLR Gaussian check: omitted covariance term
    V2 = np.linalg.inv(A / sig ** 2 + np.eye(d) / tau ** 2)
    e = 1e-3
    Ve = np.linalg.inv(np.linalg.inv(V2) - e * np.outer(X[j], X[j]) / sig ** 2)
    mue = Ve @ (X.T @ y - e * X[j] * y[j]) / sig ** 2
    b = (mue - mu) / e
    check("BLR: downweighting shifts the mean by -BIF", np.allclose(b, -bif, rtol=1e-2))

    def kl(m0, S0, m1, S1):
        S1i = np.linalg.inv(S1)
        return 0.5 * (np.trace(S1i @ S0) - d + (m1 - m0) @ S1i @ (m1 - m0) + np.log(np.linalg.det(S1) / np.linalg.det(S0)))
    Mm = np.outer(X[j], X[j]) @ V2 / sig ** 2
    pred = 0.25 * e ** 2 * np.trace(Mm @ Mm)
    check("BLR: min over constant shifts of KL = (eps^2/4) tr[(x x^T V/sigma^2)^2] + O(eps^3)",
          np.isclose(kl(mu + e * b, V2, mue, Ve), pred, rtol=1e-2), f"({kl(mu + e * b, V2, mue, Ve):.3e} vs {pred:.3e})")


# ======================================================================= stop 4: DLN
def stop4():
    # SVD derivative: A = U^T dU, C = V^T dV, E = U^T dSigma V
    S = np.diag([3.0, 1.0])
    E = np.array([[0.0, 1.0], [0.0, 0.0]])
    def svd(M):
        U, s, Vt = np.linalg.svd(M)
        sg = np.sign(np.diag(U))
        return U * sg, s, (Vt.T * sg)
    U0, s0, V0 = svd(S)
    h = 1e-6
    Up, _, Vp = svd(S + h * E)
    Um, _, Vm = svd(S - h * E)
    A = U0.T @ (Up - Um) / (2 * h)
    Cm = V0.T @ (Vp - Vm) / (2 * h)
    s1, s2 = 3.0, 1.0
    A12 = (s2 * E[0, 1] + s1 * E[1, 0]) / (s2 ** 2 - s1 ** 2)
    C12 = (s1 * E[0, 1] + s2 * E[1, 0]) / (s2 ** 2 - s1 ** 2)
    check("A_jk = (s_k E_jk + s_j E_kj)/(s_k^2 - s_j^2): S=diag(3,1), E=e1 e2^T gives -1/8",
          np.isclose(A[0, 1], A12) and np.isclose(A12, -1 / 8))
    check("C_jk = (s_j E_jk + s_k E_kj)/(s_k^2 - s_j^2) (= 3/8 with opposite sign convention)",
          np.isclose(Cm[0, 1], C12) and np.isclose(abs(C12), 3 / 8))
    check("E = A S + dS - S C", np.allclose(A @ S + np.diag(np.diag(E)) - S @ Cm, E))
    # random 3x3
    M = rng.standard_normal((3, 3))
    U0, s0, V0 = svd(M)
    dM = rng.standard_normal((3, 3))
    Up, _, Vp = svd(M + h * dM)
    Um, _, Vm = svd(M - h * dM)
    A = U0.T @ (Up - Um) / (2 * h)
    Cm = V0.T @ (Vp - Vm) / (2 * h)
    E = U0.T @ dM @ V0
    ok = True
    for jj in range(3):
        for kk in range(3):
            if jj != kk:
                sj, sk = s0[jj], s0[kk]
                ok &= np.isclose(A[jj, kk], (sk * E[jj, kk] + sj * E[kk, jj]) / (sk ** 2 - sj ** 2), atol=1e-6)
                ok &= np.isclose(Cm[jj, kk], (sj * E[jj, kk] + sk * E[kk, jj]) / (sk ** 2 - sj ** 2), atol=1e-6)
    check("SVD derivative formulas, random 3x3", ok)

    # mode-strength channel: constant delta s gives da/ds, which -> 1
    sv, a0 = 2.0, 1e-4
    t = np.linspace(0, 8, 80001)
    a = lambda s, t: s * np.exp(2 * s * t) / (np.exp(2 * s * t) - 1 + s / a0)
    adot = 2 * a(sv, t) * (sv - a(sv, t))
    integ = np.concatenate([[0], np.cumsum(0.5 * (1 / (sv - a(sv, t[1:])) + 1 / (sv - a(sv, t[:-1]))) * np.diff(t))])
    da_formula = adot * integ
    da_true = (a(sv + 1e-6, t) - a(sv - 1e-6, t)) / 2e-6
    check("speed channel, constant forcing: adot * int 1/(s-a) = da/ds", np.allclose(da_formula, da_true, atol=2e-3 * da_true.max()))
    check("da/ds -> 1 after saturation", np.isclose(da_true[-1], 1.0, atol=1e-3))

    # full DLN reweighting check (realizable dog/cat data)
    s1, s2, a0 = 4.0, 1.0, 1e-6
    Sd = np.diag([s1, s2])
    Xd = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    Yd = Xd @ Sd.T
    nd = 4
    eta = 1e-3
    steps = 10000
    p, q = 0, 1

    def run(beta):
        W1 = np.sqrt(a0) * np.eye(2)
        W2 = W1.copy()
        Sxx = (Xd.T * beta) @ Xd / nd
        Syx = (Yd.T * beta) @ Xd / nd
        out = []
        for _ in range(steps + 1):
            W = W2 @ W1
            out.append(W)
            Er = Syx - W @ Sxx
            W1, W2 = W1 + eta * W2.T @ Er, W2 + eta * Er @ W1.T
        return np.array(out)
    e = 1e-5
    bp, bm = np.ones(nd), np.ones(nd)
    bp[p] += e
    bm[p] -= e
    W0 = run(np.ones(nd))
    dW = (run(bp) - run(bm)) / (2 * e)
    tt = eta * np.arange(steps + 1)
    av = np.stack([a(s1, tt), a(s2, tt)], 1)
    adv = 2 * av * (np.array([s1, s2]) - av)
    da = (Xd[p] ** 2 / nd) * tt[:, None] * adv
    err = np.abs(np.stack([dW[:, 0, 0], dW[:, 1, 1]], 1) - da).max() / np.abs(da).max()
    check("realizable reweighting: dW_kk = (x_pk^2/n) t adot_k (time rescaling)", err < 0.02, f"(rel err {err:.3f})")
    k = int(round(4.0 / eta))
    F21 = s2 * Xd[p][1] * Xd[p][0] / nd
    plateau21 = s1 ** 2 * F21 / (s1 ** 2 - s2 ** 2)
    plateau12 = s1 * s2 * F21 / (s1 ** 2 - s2 ** 2)
    check("plateau rotation: dW21 = s1^2 F21/(s1^2-s2^2), dW12 = s1 s2 F21/(s1^2-s2^2)",
          np.isclose(dW[k, 1, 0], plateau21, rtol=0.03) and np.isclose(dW[k, 0, 1], plateau12, rtol=0.1),
          f"(fd {dW[k, 1, 0]:.4f}, {dW[k, 0, 1]:.4f}; formula {plateau21:.4f}, {plateau12:.4f})")
    xq, yq = Xd[q], Yd[q]
    rq = yq - W0[k] @ xq
    I_fd = -rq @ dW[k] @ xq
    I_plateau = -(s1 * s2) ** 2 / (nd * (s1 ** 2 - s2 ** 2)) * Xd[p][0] * Xd[p][1] * xq[0] * xq[1]
    check("plateau influence of dog on cat loss = +s1^2 s2^2/(n(s1^2-s2^2))", np.isclose(I_fd, I_plateau, rtol=0.05),
          f"(fd {I_fd:.4f}, formula {I_plateau:.4f})")
    # endpoint: realizable data -> reweighting leaves the minimizer unchanged
    check("endpoint influence -> 0 (realizable)", abs(-(Yd[q] - W0[-1] @ xq) @ dW[-1] @ xq) < 1e-4)
    # label noise: endpoint influence via weighted least squares; off-diagonal part contributes
    Yn = Xd @ Sd.T + np.array([0.0, 1.0])
    def wls(beta):
        return ((Yn.T * beta) @ Xd) @ np.linalg.inv((Xd.T * beta) @ Xd)
    dWinf = (wls(bp) - wls(bm)) / (2 * e)
    xp_ = Xd[p]
    I_full = -(Yn[p] - wls(np.ones(nd)) @ xp_) @ dWinf @ xp_
    I_diag = -(Yn[p] - wls(np.ones(nd)) @ xp_) @ np.diag(np.diag(dWinf)) @ xp_
    check("label noise: endpoint dW = e2 (1,1)/4; own-loss influence -1/2, diagonal part only -1/4",
          np.allclose(dWinf, [[0, 0], [0.25, 0.25]], atol=1e-6) and np.isclose(I_full, -0.5) and np.isclose(I_diag, -0.25))


if __name__ == "__main__":
    for f in (stop1, stop2, stop3, stop4):
        print(f"--- {f.__name__}")
        f()
    print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else ""))
    raise SystemExit(1 if FAILS else 0)
