"""Checks for every formula on the L2 slides ("Curvature at scale").

Run: python da/checks/L2_checks.py   (CPU, a few seconds). Exits non-zero on failure.
"""
import itertools
import sys

import numpy as np
import sympy as sp
import torch
from scipy.integrate import quad

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ 1. jvp, vjp, HVP routes, GN product
torch.manual_seed(0)
din, h, C, n = 4, 6, 3, 7
shapes = [(h, din), (h,), (C, h), (C,)]
sizes = [int(np.prod(s)) for s in shapes]
P = sum(sizes)
X, y = torch.randn(n, din), torch.randint(0, C, (n,))
w = torch.randn(P) * 0.7


def f(w, X=X):  # logits, (n, C)
    W1, b1, W2, b2 = [p.reshape(s) for p, s in zip(torch.split(w, sizes), shapes)]
    return torch.tanh(X @ W1.T + b1) @ W2.T + b2


def L(w):  # summed loss, as in DA1
    return torch.nn.functional.cross_entropy(f(w), y, reduction="sum")


v = torch.randn(P)
H = torch.func.hessian(L)(w)
J = torch.func.jacrev(f)(w)  # (n, C, P)
Jv_ad = torch.func.jvp(f, (w,), (v,))[1]
check("jvp = J v", torch.allclose(Jv_ad, J @ v))
u = torch.randn(n, C)
check("vjp = J^T u", torch.allclose(torch.func.vjp(f, w)[1](u)[0], torch.einsum("ncp,nc->p", J, u)))
g = torch.func.grad(L)
wr = w.clone().requires_grad_(True)
gr = torch.autograd.grad(L(wr), wr, create_graph=True)[0]
Hv_ror = torch.autograd.grad(gr @ v, wr)[0]  # reverse-over-reverse: grad of (grad L . v)
Hv_for = torch.func.jvp(g, (w,), (v,))[1]  # forward-over-reverse: jvp of the gradient
check("reverse-over-reverse Hv = H v", torch.allclose(Hv_ror, H @ v))
check("forward-over-reverse Hv = H v", torch.allclose(Hv_for, H @ v))
# Gauss-Newton: G v = sum_i J_i^T H_y,i J_i v  via one jvp, a CxC multiply, one vjp
p = torch.softmax(f(w), 1)
Hy = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]  # Hessian of CE in the logits
G = torch.einsum("ncp,ncd,ndq->pq", J, Hy, J)
Jv = torch.func.jvp(f, (w,), (v,))[1]
Gv = torch.func.vjp(f, w)[1](torch.einsum("ncd,nd->nc", Hy, Jv))[0]
check("Gv by jvp -> H_y -> vjp equals explicit G v", torch.allclose(Gv, G @ v))
check("G is PSD", torch.linalg.eigvalsh(G).min() > -1e-10)
check("H - G is not zero here (residual term)", (H - G).norm() > 1e-3 * H.norm(),
      f"|H-G|/|H| = {((H - G).norm() / H.norm()).item():.2f}")

# ------------------------------------------------------------------ 2. finite differences
x, e = sp.symbols("x epsilon", positive=True)
for fn in (sp.sin(x) * sp.exp(x / 3), sp.log(1 + x ** 2)):
    gfun = sp.diff(fn, x)
    fd = (gfun.subs(x, x + e) - gfun.subs(x, x - e)) / (2 * e)
    lead = sp.series(fd - sp.diff(gfun, x), e, 0, 3).removeO()
    check("central difference of the gradient: error = (eps^2/6) g''' + O(eps^4)",
          sp.simplify(lead - e ** 2 / 6 * sp.diff(gfun, x, 3)) == 0)
    # the gradient central difference is a uniform (box) average of the curvature on [-eps, eps]
    t = sp.symbols("t")
    box = sp.integrate(sp.diff(gfun, x).subs(x, x + t), (t, -e, e)) / (2 * e)
    check("gradient central difference = box average of H(w+tv)v", sp.simplify(box - fd) == 0)
    # the function-value second difference is a triangular-kernel average (Granziol & Juarev Thm 3.3)
    sd = (fn.subs(x, x + e) - 2 * fn + fn.subs(x, x - e)) / e ** 2
    xv, ev = 0.37, 0.21
    f2 = sp.lambdify(x, sp.diff(fn, x, 2))
    val_tri = quad(lambda tt: (1 - abs(tt) / ev) * f2(xv + tt), -ev, ev, points=[0], epsabs=1e-14, epsrel=1e-13)[0] / ev
    val_sd = float(sd.subs({x: xv, e: ev}))
    check("function second difference = triangular-kernel average of v^T H v", abs(val_tri - val_sd) < 1e-10,
          f"({val_tri:.12f} vs {val_sd:.12f})")
# optimal step and error from eps^2 c3/6 + eps_mach c1/eps
em, c1, c3 = sp.symbols("epsilon_mach c1 c3", positive=True)
E = e ** 2 * c3 / 6 + em * c1 / e
es = sp.solve(sp.diff(E, e), e)[0]
check("eps* = (3 eps_mach c1 / c3)^(1/3)", sp.simplify(es - (3 * em * c1 / c3) ** sp.Rational(1, 3)) == 0)
check("error at eps* ~ eps_mach^(2/3)", sp.simplify(sp.diff(sp.log(E.subs(e, es)), em) * em) == sp.Rational(2, 3))
fp32, bf16 = 2.0 ** -23, 2.0 ** -8
print(f"     fp32 eps_mach={fp32:.2e}: eps_mach^(1/3)={fp32 ** (1 / 3):.1e}, ^(2/3)={fp32 ** (2 / 3):.1e}")
print(f"     bf16 2^-8={bf16:.2e}: eps_mach^(1/3)={bf16 ** (1 / 3):.1e}, ^(2/3)={bf16 ** (2 / 3):.1e}")
check("orders of magnitude match Remark 3.2", 1e-3 < fp32 ** (1 / 3) < 1e-2 and 1e-5 < fp32 ** (2 / 3) < 1e-4
      and 1e-1 < bf16 ** (1 / 3) < 1 and 1e-2 < bf16 ** (2 / 3) < 1e-1)

# ------------------------------------------------------------------ 3. Hutchinson
A = rng.standard_normal((5, 5))
A = A + A.T
vals = [np.array(s) @ A @ np.array(s) for s in itertools.product([-1, 1], repeat=5)]
check("Rademacher Hutchinson is unbiased", np.isclose(np.mean(vals), np.trace(A)))
check("Rademacher variance = 2(|A|_F^2 - sum A_jj^2)",
      np.isclose(np.var(vals), 2 * (np.sum(A ** 2) - np.sum(np.diag(A) ** 2))))
Z = rng.standard_normal((400000, 5))
gv = np.einsum("ni,ij,nj->n", Z, A, Z)
check("Gaussian variance = 2|A|_F^2 (Monte Carlo, 2%)", abs(gv.var() / (2 * np.sum(A ** 2)) - 1) < 0.02)

# ------------------------------------------------------------------ 4. Lanczos and Gauss quadrature
Pm = 200
M = rng.standard_normal((Pm, Pm))
M = (M + M.T) / np.sqrt(2 * Pm)
M += np.diag(np.r_[np.zeros(Pm - 3), [3.0, 4.0, 5.0]])  # three outliers
lam, U = np.linalg.eigh(M)


def lanczos(A, q1, m):
    Q = np.zeros((len(q1), m + 1))
    a, b = np.zeros(m), np.zeros(m)
    Q[:, 0] = q1 / np.linalg.norm(q1)
    for k in range(m):
        r = A @ Q[:, k] - (b[k - 1] * Q[:, k - 1] if k else 0)
        a[k] = Q[:, k] @ r
        r -= a[k] * Q[:, k]
        r -= Q[:, :k + 1] @ (Q[:, :k + 1].T @ r)
        b[k] = np.linalg.norm(r)
        Q[:, k + 1] = r / b[k]
    return a, b, Q


m = 12
q1 = rng.standard_normal(Pm)
q1 /= np.linalg.norm(q1)
a, b, Q = lanczos(M, q1, m)
T = np.diag(a) + np.diag(b[:m - 1], 1) + np.diag(b[:m - 1], -1)
Qm = Q[:, :m]
check("Q_m^T H Q_m = T_m (tridiagonal)", np.allclose(Qm.T @ M @ Qm, T, atol=1e-12))
em_ = np.zeros(m); em_[-1] = 1
check("three-term recurrence: H Q_m = Q_m T_m + beta_m q_{m+1} e_m^T",
      np.allclose(M @ Qm, Qm @ T + b[m - 1] * np.outer(Q[:, m], em_), atol=1e-12))
check("Q_m spans the Krylov space {q1, Hq1, ..., H^{m-1} q1}",
      np.allclose(Qm @ (Qm.T @ np.linalg.matrix_power(M, m - 1) @ q1),
                  np.linalg.matrix_power(M, m - 1) @ q1, rtol=1e-8))
theta, Y = np.linalg.eigh(T)
tau2 = Y[0] ** 2
wts = (U.T @ q1) ** 2
ok = all(np.isclose(np.sum(tau2 * theta ** k), np.sum(wts * lam ** k), rtol=1e-9) for k in range(2 * m))
check("Gauss rule exact for degree <= 2m-1", ok)
d2m = abs(np.sum(tau2 * theta ** (2 * m)) / np.sum(wts * lam ** (2 * m)) - 1)
check("... and not for degree 2m", d2m > 1e-12, f"(relative error {d2m:.1e})")
check("e1^T f(T) e1 = sum tau_k^2 f(theta_k)", np.isclose(np.exp(T)[0, 0] * 0 + (Y[0] ** 2 * np.exp(theta)).sum(),
                                                          (Y @ np.diag(np.exp(theta)) @ Y.T)[0, 0]))
check("top Ritz values have converged to the outliers after m=12 (rtol 1e-4)",
      np.allclose(theta[-3:], lam[-3:], rtol=1e-4), f"{theta[-3:]} vs {lam[-3:]}")
check("interior Ritz values have not converged",
      np.min(np.abs(lam[:, None] - theta[None, 3:-3]), axis=0).max() > 1e-3)
# full Lanczos (m = P) reproduces eigh: nodes = eigenvalues, weights = (v . u_j)^2
a, b, Q = lanczos(M, q1, Pm)
T = np.diag(a) + np.diag(b[:Pm - 1], 1) + np.diag(b[:Pm - 1], -1)
th, Yf = np.linalg.eigh(T)
check("Golub-Welsch with m=P: nodes = eigh eigenvalues", np.allclose(th, lam, atol=1e-8))
check("Golub-Welsch with m=P: weights = (v.u_j)^2", np.allclose(Yf[0] ** 2, wts, atol=1e-8))
# SLQ estimator: (1/P) tr f(H) = (1/P) E[v^T f(H) v] for Rademacher v (unnormalized), here f = x^2
nv = 4000
Vs = rng.choice([-1.0, 1.0], (nv, Pm))
est = np.mean(np.einsum("ni,ij,nj->n", Vs, M @ M, Vs)) / Pm
check("Hutchinson for (1/P) tr H^2 (Monte Carlo, 3%)", abs(est / np.mean(lam ** 2) - 1) < 0.03)
# power iteration rate (lambda_2/lambda_1)^k for the eigenvector error
x0 = rng.standard_normal(Pm)
xk = x0.copy()
errs = []
for k in range(40):
    xk = M @ xk
    xk /= np.linalg.norm(xk)
    errs.append(np.linalg.norm(xk - np.sign(xk @ U[:, -1]) * U[:, -1]))
rate = (errs[-1] / errs[-11]) ** (1 / 10)
check("power iteration converges like |lambda_2/lambda_1|^k", abs(rate - lam[-2] / lam[-1]) < 0.01,
      f"(rate {rate:.3f} vs {lam[-2] / lam[-1]:.3f})")

# ------------------------------------------------------------------ 5. damped IF on bulk + outliers
s = np.r_[rng.uniform(0, 1e-3, Pm - 3), [3.0, 4.0, 5.0]]
Qr, _ = np.linalg.qr(rng.standard_normal((Pm, Pm)))
Gm = Qr @ np.diag(s) @ Qr.T
lamd = 0.1
gphi, gi = rng.standard_normal(Pm), rng.standard_normal(Pm)
exact = -gphi @ np.linalg.solve(Gm + lamd * np.eye(Pm), gi)
cphi, ci = Qr.T @ gphi, Qr.T @ gi
approx = -(cphi[:-3] @ ci[:-3]) / lamd - np.sum(cphi[-3:] * ci[-3:] / (s[-3:] + lamd))
check("damped IF = -(1/lambda) (bulk gradient dot product) + outlier terms",
      abs(approx - exact) / abs(exact) < 0.02, f"({approx:.4f} vs {exact:.4f})")

# ------------------------------------------------------------------ 6. memory arithmetic
check("1B-parameter Hessian has 1e18 entries", (1e9) ** 2 == 1e18)
print(f"     one bf16 vector of 1e11 params = {2 * 1e11 / 1e9:.0f} GB; of 1.2e11 = {2 * 1.2e11 / 1e9:.0f} GB")

print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
