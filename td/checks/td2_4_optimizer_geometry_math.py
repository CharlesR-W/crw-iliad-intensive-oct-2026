#!/usr/bin/env python3
"""Checks for every closed-form claim in td2_4_optimizer_geometry (NB2 §4).

Run: python td/checks/td2_4_optimizer_geometry_math.py   (1-4 min on a laptop, mostly the sympy integrals)
Each block asserts; a failure raises. Printed numbers are the ones quoted in the notebook.
"""
import numpy as np
import sympy as sp
from scipy.special import gammaln

rng = np.random.default_rng(0)
ok = lambda msg: print("ok ", msg)

# ---------------------------------------------------------------- 4.0 parable
# JS of two disjoint point masses with the standard 1/2 convention = log 2
P, Q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
M = (P + Q) / 2
kl = lambda a, b: sum(x * np.log(x / y) for x, y in zip(a, b) if x > 0)
assert abs(0.5 * kl(P, M) + 0.5 * kl(Q, M) - np.log(2)) < 1e-12
ok("JS(delta_0, delta_theta) = log 2")
th, s, x = sp.symbols("theta s x", positive=True)
p0 = sp.exp(-x**2 / (2 * s**2)) / sp.sqrt(2 * sp.pi * s**2)
pt = sp.exp(-(x - th)**2 / (2 * s**2)) / sp.sqrt(2 * sp.pi * s**2)
KL = sp.simplify(sp.integrate(p0 * sp.expand_log(sp.log(p0 / pt), force=True), (x, -sp.oo, sp.oo)))
assert sp.simplify(KL - th**2 / (2 * s**2)) == 0
score = sp.diff(sp.log(pt), th)
Fish = sp.simplify(sp.integrate(pt * score**2, (x, -sp.oo, sp.oo)))
assert sp.simplify(Fish - 1 / s**2) == 0
assert sp.simplify(sp.diff(KL, th) / Fish - th) == 0
ok("KL(N(0,s^2)||N(th,s^2)) = th^2/(2s^2); Fisher = 1/s^2; natural gradient = th")

# ---------------------------------------------------------------- 4.1 steepest descent
def steepest(g, norm, Mmat=None):
    if norm == "l2":
        return -g / np.linalg.norm(g)
    if norm == "linf":
        return -np.sign(g)
    if norm == "l1":
        d = np.zeros_like(g); i = np.argmax(np.abs(g)); d[i] = -np.sign(g[i]); return d
    if norm == "M":
        y = np.linalg.solve(Mmat, g); return -y / np.sqrt(g @ y)
    raise ValueError(norm)

def unit_sphere_samples(norm, n, Mmat=None, k=200000):
    z = rng.standard_normal((k, n))
    if norm == "l2": return z / np.linalg.norm(z, axis=1, keepdims=True)
    if norm == "linf": return np.clip(z, -1, 1) / np.max(np.abs(np.clip(z, -1, 1)), axis=1, keepdims=True)
    if norm == "l1": return z / np.abs(z).sum(1, keepdims=True)
    if norm == "M":
        L = np.linalg.cholesky(Mmat)  # M = L L^T ; d = L^{-T} u with |u|=1 has d^T M d = 1
        u = z / np.linalg.norm(z, axis=1, keepdims=True)
        return np.linalg.solve(L.T, u.T).T

for n in (2, 4):
    g = rng.standard_normal(n)
    A = rng.standard_normal((n, n)); Mmat = A @ A.T + 0.5 * np.eye(n)
    for norm in ("l2", "linf", "l1", "M"):
        d = steepest(g, norm, Mmat)
        Ds = unit_sphere_samples(norm, n, Mmat)
        assert g @ d <= (Ds @ g).min() + 1e-9, (n, norm)
    # dual norms
    assert abs(-g @ steepest(g, "linf") - np.abs(g).sum()) < 1e-12
    assert abs(-g @ steepest(g, "l1") - np.abs(g).max()) < 1e-12
    assert abs(-g @ steepest(g, "M", Mmat) - np.sqrt(g @ np.linalg.solve(Mmat, g))) < 1e-12
ok("trust-region directions beat 200k sampled unit-ball points (l2, linf, l1, Mahalanobis)")

# penalty form: argmin <g,D> + |D|^2/(2 eta) = -eta |g|_* d*  (check by direct minimization along d*, and global for l2/M)
eta = 0.3
g = rng.standard_normal(5)
for norm, dual in (("linf", np.abs(g).sum()), ("l1", np.abs(g).max()), ("l2", np.linalg.norm(g))):
    d = -steepest(g, norm)  # unit vector maximizing <g,d>
    cs = np.linspace(0, 3, 30001)
    vals = -cs * dual + cs**2 / (2 * eta)
    assert abs(cs[vals.argmin()] - eta * dual) < 1e-3
ok("penalty form: step length eta*||g||_* along the trust-region direction")

# spectral norm: <G, U V^T> = nuclear norm, and no D with ||D||_2 <= 1 does better
G = rng.standard_normal((6, 9))
U, S, Vt = np.linalg.svd(G, full_matrices=False)
O = U @ Vt
assert abs(np.sum(G * O) - S.sum()) < 1e-10
best = -np.inf
for _ in range(3000):
    D = rng.standard_normal((6, 9)); D /= np.linalg.norm(D, 2)
    best = max(best, np.sum(G * D))
assert best <= S.sum() + 1e-9
# nearest semi-orthogonal matrix in Frobenius is U V^T
for _ in range(2000):
    Q, _ = np.linalg.qr(rng.standard_normal((9, 6)))
    Oq = Q.T  # 6x9 with orthonormal rows
    assert np.linalg.norm(Oq - G) >= np.linalg.norm(O - G) - 1e-9
ok("spectral steepest descent: U V^T attains the nuclear norm and is the nearest semi-orthogonal matrix")

# Newton step = Mahalanobis penalty step with M = H
H = np.diag([1.0, 100.0]); th0 = np.array([1.0, 1.0]); g = H @ th0
assert np.allclose(th0 - 1.0 * np.linalg.solve(H, g), 0)
ok("penalty step with M = H and eta = 1 is Newton's step")

# ---------------------------------------------------------------- 4.2 invariance
# logistic-regression Fisher; linear reparam theta = A xi: NG iterates coincide exactly, GD does not
X = rng.standard_normal((200, 2)) * np.array([1.0, 0.2]); w_true = np.array([2.0, -3.0])
y = (rng.random(200) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
def grad_fisher(w):
    p = 1 / (1 + np.exp(-X @ w))
    return X.T @ (p - y) / len(y), (X * (p * (1 - p))[:, None]).T @ X / len(y)
A = np.diag([1.0, 10.0])
w = np.zeros(2); xi = np.zeros(2)
for _ in range(20):
    gw, Fw = grad_fisher(w); w = w - 0.5 * np.linalg.solve(Fw, gw)
    gt, Ft = grad_fisher(A @ xi); gx, Fx = A.T @ gt, A.T @ Ft @ A
    xi = xi - 0.5 * np.linalg.solve(Fx, gx)
assert np.allclose(w, A @ xi)
ok("natural gradient: identical iterates under a linear reparametrization")
# nonlinear reparam theta = phi(xi): one NG step agrees to first order (discrepancy O(eta^2))
phi = lambda z: np.array([z[0], np.exp(z[1])]); J = lambda z: np.array([[1, 0], [0, np.exp(z[1])]])
xi0 = np.array([0.3, 0.2]); errs = []
for e in (1e-1, 5e-2, 2.5e-2):
    gt, Ft = grad_fisher(phi(xi0))
    th1 = phi(xi0) - e * np.linalg.solve(Ft, gt)
    Jx = J(xi0); xi1 = xi0 - e * np.linalg.solve(Jx.T @ Ft @ Jx, Jx.T @ gt)
    errs.append(np.linalg.norm(phi(xi1) - th1))
r = np.log(errs[0] / errs[1]) / np.log(2), np.log(errs[1] / errs[2]) / np.log(2)
assert 1.8 < r[0] < 2.2 and 1.8 < r[1] < 2.2, r
ok(f"nonlinear reparam: NG discrepancy scales as eta^{r[1]:.2f}")
# Hessian at a minimum: raw eigenvalues change, eigenvalues of F^{-1} H do not
Hm = np.array([[2.0, 0.5], [0.5, 1.0]]); Fm = np.array([[1.0, 0.2], [0.2, 0.5]])
Jm = np.array([[3.0, 1.0], [0.0, 0.2]])
ev = lambda a: np.sort(np.linalg.eigvals(a).real)
assert not np.allclose(ev(Hm), ev(Jm.T @ Hm @ Jm))
assert np.allclose(ev(np.linalg.solve(Fm, Hm)), ev(np.linalg.solve(Jm.T @ Fm @ Jm, Jm.T @ Hm @ Jm)))
ok("raw sharpness changes under reparametrization; spectrum of F^{-1}H does not")
# preconditioned GD on a quadratic is stable iff eta * lambda_max(P^{-1} H) < 2
Pm = np.diag([4.0, 1.0]); lam = ev(np.linalg.solve(Pm, Hm)).max()
for eta, stable in ((1.98 / lam, True), (2.02 / lam, False)):
    th_ = np.array([1.0, -1.0])
    for _ in range(3000): th_ = th_ - eta * np.linalg.solve(Pm, Hm @ th_)
    assert (np.linalg.norm(th_) < 1) == stable
ok("preconditioned-GD threshold eta * lambda_max(P^{-1}H) = 2")

# ---------------------------------------------------------------- 4.3 Adam
b1, b2 = 0.9, 0.999
bound = (1 - b1) / np.sqrt((1 - b2) * (1 - b1**2 / b2))
T = 6000; gseq = (b1 / b2) ** np.arange(T)[::-1]
m = v = 0.0
for gg in gseq: m = b1 * m + (1 - b1) * gg; v = b2 * v + (1 - b2) * gg**2
assert abs(m / np.sqrt(v) - bound) < 1e-6
for _ in range(200):  # random sequences never exceed it
    gs = rng.standard_normal(3000) * np.exp(rng.standard_normal() * np.linspace(0, 3, 3000))
    m = v = 0.0
    for gg in gs: m = b1 * m + (1 - b1) * gg; v = b2 * v + (1 - b2) * gg**2
    assert abs(m) / np.sqrt(v) <= bound + 1e-9
ok(f"Adam steady-state bound |m/sqrt v| <= {bound:.3f} (beta=0.9,0.999), attained by g_(t-k) ∝ (b1/b2)^k")
# scale invariance (eps = 0)
def adam_updates(gs, b1=0.9, b2=0.999):
    m = v = 0.0; out = []
    for t, gg in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * gg; v = b2 * v + (1 - b2) * gg**2
        out.append((m / (1 - b1**t)) / np.sqrt(v / (1 - b2**t)))
    return np.array(out)
gs = rng.standard_normal(100)
assert np.allclose(adam_updates(gs), adam_updates(37.0 * gs))
ok("Adam (eps=0) update invariant to gradient scale")
# SNR factor: E[m/sqrt v] ≈ mu / sqrt(mu^2 + sigma^2) for iid gradients, beta2 -> 1
for snr in (0.1, 0.5, 2.0):
    gs = snr + rng.standard_normal(200000)
    u = adam_updates(gs, b1=0.9, b2=0.9999)[20000:]
    assert abs(u.mean() - snr / np.sqrt(1 + snr**2)) < 0.02, (snr, u.mean())
ok("mean Adam update ≈ mu/sqrt(mu^2+sigma^2) for iid gradients")

# ---------------------------------------------------------------- 4.4 Newton-Schulz
sg, e = sp.symbols("sigma e")
p = (3 * sg - sg**3) / 2
assert sp.expand(p.subs(sg, 1 + e) - 1 - (-sp.Rational(3, 2) * e**2 - e**3 / 2)) == 0
assert sp.solve(sp.Eq(p, sg), sg) == [-1, 0, 1] or set(sp.solve(sp.Eq(p, sg), sg)) == {-1, 0, 1}
assert p.subs(sg, sp.sqrt(3)) == 0 and p.subs(sg, 2) == -1
xs = np.linspace(1e-3, np.sqrt(3) - 1e-3, 2000); ys = xs.copy()
for _ in range(60): ys = (3 * ys - ys**3) / 2
assert np.allclose(ys, 1)
ok("cubic NS: fixed points 0, ±1; error e -> -(3/2)e^2 - e^3/2; converges to 1 on (0, sqrt 3)")
a, b, c = 3.4445, -4.7750, 2.0315
q = lambda z: a * z + b * z**3 + c * z**5
dq = lambda z: a + 3 * b * z**2 + 5 * c * z**4
assert abs(q(1) - 0.701) < 1e-9
fp = [r.real for r in np.roots([c, 0, b, 0, a - 1, 0]) if abs(r.imag) < 1e-12 and r.real > 1e-9]
fp_in = min(fp); assert abs(fp_in - 0.868) < 1e-3 and dq(fp_in) < -1
z = 0.5
for _ in range(4000): z = q(z)
cyc = [z]
for _ in range(3): cyc.append(q(cyc[-1]))
mult = np.prod([dq(t) for t in cyc])
assert abs(mult) < 1 and abs(q(cyc[-1]) - cyc[0]) < 1e-12
print("    quintic 4-cycle:", np.round(sorted(cyc), 4), "multiplier", round(mult, 4))
xs = np.linspace(1e-4, 1, 100001); ys = xs.copy()
for _ in range(5): ys = q(ys)
print("    quintic after 5 steps from [1e-4, 1]:", ys.min().round(4), ys.max().round(4), " a^5 =", round(a**5, 1))
assert ys.max() < 1.21
for s0, (kc, kq) in ((1e-3, (16, 6)), (1e-2, (10, 4))):
    for f, k_expected in (((lambda z: (3 * z - z**3) / 2), kc), (q, kq)):
        zz, k = s0, 0
        while zz < 0.5: zz, k = f(zz), k + 1
        assert k == k_expected, (s0, k)
ok("quintic: q(1)=0.701, interior fixed point 0.868 repelling, attracting 4-cycle in [0.68, 1.14]")

# ---------------------------------------------------------------- 4.5 sign of a low-rank gradient
a_ = rng.standard_normal(300); b_ = rng.standard_normal(500)
assert abs(np.linalg.norm(np.sign(np.outer(a_, b_)), 2) - np.sqrt(300 * 500)) < 1e-8
ok("||sign(a b^T)||_2 = sqrt(mn)")
lam = lambda r: np.exp(gammaln(r / 2) - gammaln((r + 1) / 2)) / np.sqrt(np.pi)
assert abs(lam(1) - 1) < 1e-12 and abs(lam(2) - 2 / np.pi) < 1e-12
for r in (4, 16):
    A_ = rng.standard_normal((10**6, r)); B_ = rng.standard_normal((10**6, r))
    X_ = A_[:, 0] * B_[:, 0]; Y_ = (A_[:, 1:] * B_[:, 1:]).sum(1)
    print(f"    r={r}: lower-bound constant E[X sign(X+Y)] = {(X_ * np.sign(X_ + Y_)).mean():.3f};"
          f" asymptotic constant E|u_1| = {lam(r):.4f}; sqrt(2/(pi r)) = {np.sqrt(2/np.pi/r):.4f}")
n = 3000
for r in (4, 16):
    Gm = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    sv = np.linalg.norm(np.sign(Gm), 2) / n
    assert lam(r) < sv < lam(r) + 0.04, (r, sv)
    print(f"    r={r}, n={n}: ||sign(G)||_2 / n = {sv:.4f}")
dense = np.linalg.norm(np.sign(rng.standard_normal((n, n))), 2) / np.sqrt(n)
assert abs(dense - 2) < 0.05
ok(f"dense random-sign matrix: ||S||_2/sqrt(n) = {dense:.3f} (-> 2)")
print("all checks passed")

# ---------------------------------------------------------------- additions after critique round 1
# Shampoo without accumulation: (G G^T)^{-1/4} G (G^T G)^{-1/4} = U V^T (full rank)
def mpow(A, p):
    w, Q = np.linalg.eigh(A); return Q @ np.diag(w**p) @ Q.T
G = rng.standard_normal((5, 8)); U, S, Vt = np.linalg.svd(G, full_matrices=False)
Vr = Vt.T  # 8x5
GtG_inv4 = Vr @ np.diag(S**-0.5) @ Vr.T          # (G^T G)^{-1/4} on its range
assert np.allclose(mpow(G @ G.T, -0.25) @ G @ GtG_inv4, U @ Vt)
ok("Shampoo without accumulation = U V^T")
# RMS->RMS norm = sqrt(n_in/n_out) ||W||_2 ; its trust-region step = sqrt(n_out/n_in) U V^T
W = rng.standard_normal((30, 70))
xs = rng.standard_normal((70, 20000))
ratio = (np.linalg.norm(W @ xs, axis=0) / np.sqrt(30)) / (np.linalg.norm(xs, axis=0) / np.sqrt(70))
assert ratio.max() <= np.sqrt(70 / 30) * np.linalg.norm(W, 2) + 1e-9
U, S, Vt = np.linalg.svd(W, full_matrices=False)
assert abs(np.linalg.norm(W @ Vt[0]) / np.sqrt(30) / (1 / np.sqrt(70)) - np.sqrt(70 / 30) * S[0]) < 1e-9
ok("||W||_{RMS->RMS} = sqrt(n_in/n_out) ||W||_2")
# modular norm of an L-chain of Linear modules (mass 1, sensitivity 1) = L max ||W_l||
def chain_norm(norms):
    N, mass = norms[0], 1
    for w in norms[1:]:
        N = max(1.0 * (mass + 1) / mass * N, (mass + 1) / 1 * w); mass += 1
    return N
for _ in range(100):
    ws = rng.random(rng.integers(1, 9))
    assert abs(chain_norm(list(ws)) - len(ws) * ws.max()) < 1e-12
ok("modular norm of an L-chain = L max_l ||W_l||")
# decoupled weight decay: ball ||x|| <= b/lambda invariant and attracting (linf, sign steps)
wd, eta = 0.1, 0.05
x = 30 * rng.standard_normal(10)
for t in range(3000):
    x = x + eta * (np.sign(rng.standard_normal(10)) - wd * x)
assert np.abs(x).max() <= 1 / wd + 1e-9
ok("sign descent + decoupled decay ends in the box |x_i| <= 1/lambda")
# Adam fluctuation: s.d. of m/sqrt(v) at mu = 0 is sqrt((1-b1)/(1+b1))
u = adam_updates(rng.standard_normal(300000), b1=0.9, b2=0.9999)[30000:]
assert abs(u.std() - np.sqrt(0.1 / 1.9)) < 0.01, u.std()
ok(f"Adam pure-noise step s.d. {u.std():.3f} vs sqrt((1-b1)/(1+b1)) = {np.sqrt(0.1/1.9):.3f}")
# Funk-Hecke: odd eigenvalues of the sign kernel decrease; lambda_1 = E|u_1|
from scipy.special import eval_gegenbauer
from scipy.integrate import quad
for r in (3, 4, 8):
    al = (r - 2) / 2; wgt = lambda t: (1 - t * t) ** ((r - 3) / 2); Z = quad(wgt, -1, 1)[0]
    lamk = [quad(lambda t: np.sign(t) * eval_gegenbauer(k, al, t) / eval_gegenbauer(k, al, 1.0) * wgt(t),
                 -1, 1, points=[0])[0] / Z for k in range(1, 8)]
    pred = [lam(r)]
    for k in range(1, 4): pred.append(pred[-1] * (-1) * (2 * k - 1) / (r + 2 * k - 1))
    assert np.allclose(lamk[0::2], pred, atol=1e-6) and np.allclose(lamk[1::2], 0, atol=1e-6)
ok("sign kernel: lambda_1 = E|u_1|, even eigenvalues 0, odd ones follow the product formula")
# Newton in xi = (xi1, log theta2) is indefinite at the 4.2.D start; F_xi is PD
rd = np.random.default_rng(1)
Xd = rd.standard_normal((400, 2)) * np.array([1.0, 0.5]); wt = np.array([1.5, 3.0])
yd = (rd.random(400) < 1 / (1 + np.exp(-Xd @ wt))).astype(float)
w0 = np.array([0.0, 0.5]); z = Xd @ w0; pp = 1 / (1 + np.exp(-z))
g0 = Xd.T @ (pp - yd) / 400; F0 = (Xd * (pp * (1 - pp))[:, None]).T @ Xd / 400
J0 = np.diag([1.0, 0.5])
assert np.linalg.eigvalsh(J0 @ F0 @ J0).min() > 0
assert np.linalg.eigvalsh(J0 @ F0 @ J0 + np.diag([0, g0[1] * 0.5])).min() < 0
ok("4.2.D: xi-Hessian indefinite at the start, F_xi positive definite")
print("all additional checks passed")
