#!/usr/bin/env python3
"""Checks for the closed-form claims in build_td2_6_curvature.py (NB2 §6).

Run: python checks/td2_6_curvature_check.py   (sympy, numpy, torch; a few seconds)
"""
import numpy as np
import sympy as sp
import torch

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

# 6.1.B.1  softmax CE gradient and Hessian
C = 4
z = sp.symbols("z0:%d" % C)
y = 1
p = [sp.exp(zi) / sum(sp.exp(zj) for zj in z) for zi in z]
l = -sp.log(p[y])
for c in range(C):
    assert sp.simplify(sp.diff(l, z[c]) - (p[c] - (1 if c == y else 0))) == 0
Hs = sp.Matrix(C, C, lambda i, j: sp.diff(l, z[i], z[j]))
A = sp.Matrix(C, C, lambda i, j: (p[i] if i == j else 0) - p[i] * p[j])
assert sp.simplify(Hs - A) == sp.zeros(C, C)
print("6.1.B.1 grad = p - e_y, Hessian = diag(p) - pp^T: ok")

# 6.1.B.2-4  second-moment form, PSD, rank C-1, trace, lambda_max <= max p, Fisher identity, near-interpolation form
for _ in range(20):
    K = 6
    zz = rng.normal(size=K) * 3
    pp = np.exp(zz - zz.max()); pp /= pp.sum()
    AA = np.diag(pp) - np.outer(pp, pp)
    E = np.eye(K)
    assert np.allclose(AA, sum(pp[c] * np.outer(E[c] - pp, E[c] - pp) for c in range(K)))
    ev = np.linalg.eigvalsh(AA)
    assert abs(ev[0]) < 1e-12 and ev[1] > 0
    assert abs(np.trace(AA) - (1 - pp @ pp)) < 1e-12 and ev[-1] <= pp.max() + 1e-12
    yy = rng.integers(K)
    assert np.allclose(pp - E[yy], sum(pp[c] * (E[c] - E[yy]) for c in range(K) if c != yy))
# near interpolation: A - eps * sum q_c (e_c - e_y)(e_c - e_y)^T = O(eps^2)
for eps in (1e-2, 1e-3):
    K, yy = 5, 2
    q = rng.random(K); q[yy] = 0; q /= q.sum()
    pp = eps * q; pp[yy] = 1 - eps
    AA = np.diag(pp) - np.outer(pp, pp)
    E = np.eye(K)
    approx = eps * sum(q[c] * np.outer(E[c] - E[yy], E[c] - E[yy]) for c in range(K) if c != yy)
    print(f"6.1.B.4 eps={eps:g}: |A - leading| / eps^2 = {np.abs(AA - approx).max() / eps**2:.2f} (bounded)")
print("6.1.B.2-4: ok")

# 6.1.A.3  scalar net: H_res eigenvalues +-r, det H on the balanced line
w1, w2, s, a = sp.symbols("w1 w2 s a", real=True)
L = (w1 * w2 - s) ** 2 / 2
H = sp.hessian(L, (w1, w2))
HGN = sp.Matrix([[w2**2, w1 * w2], [w1 * w2, w1**2]])
r = w1 * w2 - s
assert sp.simplify(H - HGN - r * sp.Matrix([[0, 1], [1, 0]])) == sp.zeros(2, 2)
assert sp.factor(sp.expand(H.subs({w1: a, w2: a}).det() - (s - a**2) * (3 * a**2 - s))) == 0
print("6.1.A.3 H_res = r [[0,1],[1,0]], det H(a,a) = (s - a^2)(3a^2 - s): ok")

# 6.2.A.1  nonzero spectra of J^T A J and A^1/2 J J^T A^1/2, eigenvector map, rank n(C-1)
n, C, P = 3, 4, 50
J = rng.normal(size=(n * C, P))
Ab = np.zeros((n * C, n * C))
for i in range(n):
    zz = rng.normal(size=C); pp = np.exp(zz) / np.exp(zz).sum()
    Ab[C * i:C * i + C, C * i:C * i + C] = np.diag(pp) - np.outer(pp, pp)
w, V = np.linalg.eigh(Ab)
Ah = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
G = J.T @ Ab @ J / n
K_ = Ah @ J @ J.T @ Ah / n
e1, e2 = np.sort(np.linalg.eigvalsh(G))[::-1], np.sort(np.linalg.eigvalsh(K_))[::-1]
assert np.allclose(e1[: n * (C - 1)], e2[: n * (C - 1)]) and np.sum(e1 > 1e-9) == n * (C - 1)
lam, vv = np.linalg.eigh(K_)
u = J.T @ Ah @ vv[:, -1] / np.sqrt(n * lam[-1])
assert np.allclose(G @ u, lam[-1] * u) and abs(np.linalg.norm(u) - 1) < 1e-10
print("6.2.A.1 shared nonzero spectrum, eigenvector map, rank n(C-1): ok")

# 6.2.A.4  balanced linear net at its global minimum: Hessian nonzero spectrum = {s_a + s_b}, S = 2 s_1
torch.manual_seed(0)
m, h, d, rk = 4, 2, 5, 2                     # rank-deficient target: r = 2 < m = 4 < d = 5
T = torch.randn(m, rk) @ torch.randn(rk, d)
U, sv, Vh = torch.linalg.svd(T, full_matrices=False)
U, sv, Vh = U[:, :rk], sv[:rk], Vh[:rk]
R = torch.linalg.qr(torch.randn(h, h))[0]
W1 = R @ torch.diag(sv.sqrt()) @ Vh
W2 = U @ torch.diag(sv.sqrt()) @ R.T
assert torch.allclose(W2 @ W1, T)
th = torch.cat([W1.flatten(), W2.flatten()])
Lf = lambda t: 0.5 * ((T - t[h * d:].reshape(m, h) @ t[: h * d].reshape(h, d)) ** 2).sum()
eH = torch.linalg.eigvalsh(torch.func.hessian(Lf)(th)).flip(0).numpy()
pred = sorted([(sv[i] + sv[j]).item() for i in range(rk) for j in range(rk)]
              + [sv[i].item() for i in range(rk)] * (d - rk) + [sv[j].item() for j in range(rk)] * (m - rk), reverse=True)
assert np.allclose(eH[eH > 1e-9], pred) and abs(eH[0] - 2 * sv[0].item()) < 1e-10
print(f"6.2.A.4 linear-net minimum: nonzero Hessian spectrum = s_a + s_b etc., S = 2 s_1 = {eH[0]:.4f}: ok")

# 6.3.A.4  zero last layer: H_GN = Abar kron M2
C, hdim, nn = 4, 5, 30
Hf = rng.normal(size=(nn, hdim)) + 1.0
Ht = np.c_[Hf, np.ones(nn)]
M2 = Ht.T @ Ht / nn
Abar = (np.eye(C) - np.ones((C, C)) / C) / C
Jlast = [np.kron(np.eye(C), Ht[i][None, :]) for i in range(nn)]     # z_i = W h_i + b, vec row-major
Gce = sum(Ji.T @ Abar @ Ji for Ji in Jlast) / nn
assert np.allclose(Gce, np.kron(Abar, M2))
eg = np.sort(np.linalg.eigvalsh(Gce))[::-1]
assert np.allclose(eg[: C - 1], np.linalg.eigvalsh(M2)[-1] / C) and eg[C - 1] < eg[0] * 0.999
print("6.3.A.4 zero last layer: H_GN = Abar kron M2, top eigenvalue lambda_max(M2)/C with multiplicity C-1: ok")

# 6.4  Delta L along an eigendirection; valley stationary variance
eta, lam_, g, x, sig = sp.symbols("eta lambda g x sigma", positive=True)
assert sp.simplify((-eta * g**2 + eta**2 * lam_ * g**2 / 2) - (-eta * g**2 * (1 - eta * lam_ / 2))) == 0
dDom = sp.Rational(1, 2) * lam_ * ((1 - eta * lam_) * x) ** 2 - sp.Rational(1, 2) * lam_ * x**2
assert sp.simplify(dDom + eta * lam_**2 * x**2 * (2 - eta * lam_) / 2) == 0
v = sp.symbols("v")
vs = sp.solve(sp.Eq(v, (1 - eta * lam_) ** 2 * v + eta**2 * sig**2), v)[0]
assert sp.simplify(vs - eta * sig**2 / (lam_ * (2 - eta * lam_))) == 0
print("6.4.A Delta L = -eta g^2 (1 - eta lam/2); stationary E x^2 = eta sigma^2 / (a (2 - eta a)): ok")

# 6.5.A  scalar-net barriers
al, A_, B_ = sp.symbols("alpha a b", positive=True)
prod_same = ((1 - al) * A_ + al * B_) * ((1 - al) * s / A_ + al * s / B_)
assert sp.simplify(prod_same - s * (1 + al * (1 - al) * (A_ - B_) ** 2 / (A_ * B_))) == 0
bar_same = sp.simplify(((prod_same - s) ** 2 / 2).subs(al, sp.Rational(1, 2)))
assert sp.simplify(bar_same - s**2 * (A_ - B_) ** 4 / (32 * A_**2 * B_**2)) == 0
prod_opp = ((1 - al) * A_ - al * B_) * ((1 - al) * s / A_ - al * s / B_)
assert sp.simplify(prod_opp - s / (A_ * B_) * (al * (A_ + B_) - A_) * (al * (A_ + B_) - B_)) == 0
assert sp.simplify(prod_opp.subs(al, sp.Rational(1, 2)) + s * (A_ - B_) ** 2 / (4 * A_ * B_)) == 0
# the max of (prod - s)^2/2 is at alpha = 1/2 (prod is a convex quadratic in alpha with min there): numeric spot check
for aa, bb in ((1.0, 1.0), (2.0, 0.5), (0.3, 1.7)):
    al_ = np.linspace(0, 1, 100001)
    pr = ((1 - al_) * aa - al_ * bb) * ((1 - al_) / aa - al_ / bb)
    assert abs((0.5 * (pr - 1) ** 2).max() - 0.5 * ((aa + bb) ** 2 / (4 * aa * bb)) ** 2) < 1e-8
print("6.5.A barriers: same branch s^2(a-b)^4/(32a^2b^2) =", float(bar_same.subs({s: 1, A_: 1.1, B_: 0.9})),
      "; opposite branches (s^2/2)((a+b)^2/(4ab))^2: ok")
# 6.5.A.5  imbalance: Theta at the minimum = sqrt(d^2 + 4 s^2)
dd, ss = sp.symbols("d s", positive=True)
w1sq = (-dd + sp.sqrt(dd**2 + 4 * ss**2)) / 2
w2sq = w1sq + dd
assert sp.simplify(w1sq * w2sq - ss**2) == 0 and sp.simplify(w1sq + w2sq - sp.sqrt(dd**2 + 4 * ss**2)) == 0
print("6.5.A.5 Theta at the minimum = sqrt(d^2 + 4 s^2): ok")

# 6.3.A.3  range split: per class, span{m_ck}_k = span{mbar_c} + (C-2)-dim deviations
C = 5
for c in range(3):
    w = rng.random(C); m_ = rng.normal(size=(C, 30))
    m_[c] = -(w[np.arange(C) != c, None] * m_[np.arange(C) != c]).sum(0) / w[c]    # enforce sum_k w_ck m_ck = 0
    assert np.linalg.matrix_rank(m_) == C - 1
    others = np.arange(C) != c
    mbar = (w[others, None] * m_[others]).sum(0) / w[others].sum()
    assert np.allclose(np.cross(m_[c][:3], mbar[:3]), 0, atol=1e-9)                # m_cc parallel to mbar_c
    dev = m_[others] - mbar
    assert np.linalg.matrix_rank(dev) == C - 2
print("6.3.A.3 class direction + (C-2) deviations per class: ok")
# 6.5.A.3  fork disagreement for the linearized noisy escape: P(opposite) = arccos(rho)/pi
s_, sig_, a0_, t_ = 0.5, 0.3, 0.1, 1.3
N = 400000
beta = a0_ * rng.standard_normal(N) + np.sqrt(sig_**2 / (2 * s_) * (1 - np.exp(-2 * s_ * t_))) * rng.standard_normal(N)
fut = np.sqrt(sig_**2 / (2 * s_) * np.exp(-2 * s_ * t_))
xa, xb = beta + fut * rng.standard_normal(N), beta + fut * rng.standard_normal(N)
q_ = sig_**2 / (2 * s_)
rho_ = 1 - q_ / (a0_**2 + q_) * np.exp(-2 * s_ * t_)
assert abs((np.sign(xa) != np.sign(xb)).mean() - np.arccos(rho_) / np.pi) < 3e-3
print("6.5.A.3 P(opposite) = arccos(rho)/pi (Monte Carlo): ok")
print("all checks passed")
