"""Symbolic and numerical checks for every closed form stated in td1_linear_networks.

Run: python td/checks/td1_linear_networks_math.py   (prints OK lines, asserts on failure)
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.linalg import sqrtm, expm

rng = np.random.default_rng(0)
ok = lambda m: print("OK ", m)

# ---------------------------------------------------------------- 1. loss = 1/2||S - W||^2 + const under whitening
nx, ny, n = 4, 3, 50
X = rng.normal(size=(nx, n)); X = np.linalg.cholesky(np.linalg.inv(X @ X.T / n)).T @ X  # whiten: XX^T/n = I
assert np.allclose(X @ X.T / n, np.eye(nx))
Y = rng.normal(size=(ny, n)); S = Y @ X.T / n
W = rng.normal(size=(ny, nx))
L = 0.5 * np.mean(np.sum((Y - W @ X) ** 2, 0))
assert np.isclose(L, 0.5 * np.sum((S - W) ** 2) + 0.5 * np.mean(np.sum(Y**2, 0)) - 0.5 * np.sum(S**2))
ok("whitened loss = 1/2||Sigma_yx - W||_F^2 + const")

# ---------------------------------------------------------------- 2. conservation (continuous) and discrete drift
h = 5
W1 = rng.normal(size=(h, nx)); W2 = rng.normal(size=(ny, h))
E = S - W2 @ W1
dW1, dW2 = W2.T @ E, E @ W1.T
dC = dW1 @ W1.T + W1 @ dW1.T - dW2.T @ W2 - W2.T @ dW2
assert np.allclose(dC, 0); ok("d/dt (W1W1^T - W2^TW2) = 0")
eta = 1e-2
W1n, W2n = W1 + eta * dW1, W2 + eta * dW2
dCd = (W1n @ W1n.T - W2n.T @ W2n) - (W1 @ W1.T - W2.T @ W2)
assert np.allclose(dCd, eta**2 * (W2.T @ E @ E.T @ W2 - W1 @ E.T @ E @ W1.T)); ok("discrete GD drift = eta^2 (W2^T E E^T W2 - W1 E^T E W1^T)")

# ---------------------------------------------------------------- 3. scalar mode ODE with imbalance
a, s, d, t, a0, b, c = sp.symbols('a s d t a0 b c', real=True)
# a=bc, d=c^2-b^2 => b^2+c^2 = sqrt(4a^2+d^2)
assert sp.simplify((b**2 + c**2)**2 - (4 * (b * c)**2 + (c**2 - b**2)**2)) == 0; ok("b^2+c^2 = sqrt(4a^2+d^2)")
# closed form t(a) for general d
R = sp.sqrt(4 * s**2 + d**2)
F = sp.log((4 * s * a + d**2 + R * sp.sqrt(4 * a**2 + d**2)) / (2 * s - 2 * a)) / R
expr = sp.diff(F, a) - 1 / ((s - a) * sp.sqrt(4 * a**2 + d**2))
for vals in [(1.0, 0.3, 0.2), (2.5, 1.7, 0.01), (0.7, -2.0, -0.3), (3, 0, 1.2)]:
    sv, dv, av = vals
    assert abs(float(expr.subs({s: sv, d: dv, a: av}))) < 1e-10
ok("t(a)/tau = F(a)-F(a0), F = ln[(4sa+d^2+sqrt(4s^2+d^2)sqrt(4a^2+d^2))/(2s-2a)]/sqrt(4s^2+d^2)")
# balanced logistic
abal = s * sp.exp(2 * s * t) / (sp.exp(2 * s * t) - 1 + s / a0)
assert sp.simplify(sp.diff(abal, t) - 2 * abal * (s - abal)) == 0 and sp.simplify(abal.subs(t, 0) - a0) == 0
ok("balanced logistic solution (t in units of tau)")
# half time
th = sp.log(s / a0 - 1) / (2 * s)
assert sp.simplify(abal.subs(t, th) - s / 2) == 0; ok("t_1/2 = (tau/2s) ln(s/a0 - 1)")
# numerical: ODE vs closed form, general d
for sv, dv, a0v in [(2.0, 0.0, 1e-3), (2.0, 0.5, 1e-3), (1.0, 8.0, 0.0), (1.0, -3.0, -0.5)]:
    sol = solve_ivp(lambda tt, y: np.sqrt(4 * y**2 + dv**2) * (sv - y), [0, 12], [a0v], rtol=1e-11, atol=1e-13, dense_output=True)
    Ff = sp.lambdify(a, F.subs({s: sv, d: dv}))
    for tt in [0.5, 1.0, 3.0]:
        av = sol.sol(tt)[0]
        if abs(sv - av) > 1e-6 and (dv != 0 or av > 0):
            assert abs(Ff(av) - Ff(a0v) - tt) < 1e-6, (sv, dv, tt)
ok("ODE integration matches closed-form t(a) for several (s,d,a0)")

# ---------------------------------------------------------------- 4. lazy limit: relative kernel change
# kernel for a mode: k = b^2+c^2 = sqrt(4a^2+d^2); from a=0 to a=s: k goes |d| -> sqrt(4s^2+d^2)
for dv in [10, 100]:
    rel = np.sqrt(4 + dv**2) / dv - 1  # s=1
    assert abs(rel - 2 / dv**2) < 2 / dv**4 * 3
ok("relative kernel change from a=0 to a=s is ~2 s^2/d^2 for |d|>>s")

# ---------------------------------------------------------------- 5. depth N balanced
N = sp.symbols('N', positive=True)
w = sp.symbols('w', positive=True)
# a = w^N, tau w' = w^{N-1}(s-a) => tau a' = N w^{N-1} w^{N-1}(s-a) = N a^{2-2/N}(s-a)
lhs = N * w**(N - 1) * w**(N - 1) * (s - w**N)
rhs = N * (w**N)**(2 - 2 / N) * (s - w**N)
assert sp.simplify(sp.powsimp(sp.expand_power_base(lhs - rhs, force=True), force=True)) == 0
ok("depth-N balanced: tau a' = N a^(2-2/N)(s-a)")
# plateau scaling: escape time to a=s/2 from a0 small, N>=3 ~ tau a0^{-(1-2/N)}/(s(N-2))
for Nv in [3, 4, 6]:
    sv = 1.0
    for a0v in [1e-3, 1e-4]:
        T = quad(lambda x: 1 / (Nv * x**(2 - 2 / Nv) * (sv - x)), a0v, sv / 2, limit=200)[0]
        pred = a0v**(-(1 - 2 / Nv)) / (sv * (Nv - 2))
        assert abs(T / pred - 1) < 0.25, (Nv, a0v, T, pred)
ok("N>=3 escape time ~ tau a0^-(1-2/N) / (s(N-2)) (leading order)")

# ---------------------------------------------------------------- 6. Hessian at origin: strict (N=2) vs zero (N=3)
bs = sp.symbols('b1:4', real=True)
L2 = sp.Rational(1, 2) * (s - bs[0] * bs[1])**2
H2 = sp.hessian(L2, bs[:2]).subs({bs[0]: 0, bs[1]: 0})
assert H2 == sp.Matrix([[0, -s], [-s, 0]])
L3 = sp.Rational(1, 2) * (s - bs[0] * bs[1] * bs[2])**2
assert sp.hessian(L3, bs).subs({x: 0 for x in bs}) == sp.zeros(3)
ok("mode Hessian at 0: [[0,-s],[-s,0]] for N=2, zero for N=3")

# ---------------------------------------------------------------- 7. end-to-end NTK and balanced square roots
E = S - W2 @ W1
Wdot = W2 @ dW1 + dW2 @ W1
assert np.allclose(Wdot, W2 @ W2.T @ E + E @ W1.T @ W1); ok("W' = W2W2^T E + E W1^T W1")
# tangent kernel of f(x)=W2W1x: Theta(x,x') = (x.x') W2W2^T + (x^T W1^T W1 x') I
x1, x2 = rng.normal(size=nx), rng.normal(size=nx)
J = lambda x: np.hstack([np.kron(W2, x[None, :]), np.kron(np.eye(ny), (W1 @ x)[None, :])])  # d f / d (vec W1 row-major, vec W2 row-major)
# check J by finite differences
eps = 1e-6
def f(th, x):
    A = th[:h * nx].reshape(h, nx); B = th[h * nx:].reshape(ny, h); return B @ A @ x
th0 = np.concatenate([W1.ravel(), W2.ravel()])
Jfd = np.stack([(f(th0 + eps * e, x1) - f(th0 - eps * e, x1)) / (2 * eps) for e in np.eye(th0.size)], 1)
assert np.allclose(Jfd, J(x1), atol=1e-6)
Theta = J(x1) @ J(x2).T
assert np.allclose(Theta, (x1 @ x2) * W2 @ W2.T + (x1 @ W1.T @ W1 @ x2) * np.eye(ny)); ok("NTK Theta(x,x') = (x.x')W2W2^T + (x^T W1^T W1 x') I")
# balanced: build W1, W2 with W1W1^T = W2^TW2: W1 = Q D^{1/2} V^T ... take W = U D V^T, W2 = U D^{1/2} Q^T, W1 = Q D^{1/2} V^T
r = 3
Uq, _ = np.linalg.qr(rng.normal(size=(ny, r))); Vq, _ = np.linalg.qr(rng.normal(size=(nx, r))); Q, _ = np.linalg.qr(rng.normal(size=(h, r)))
D = np.diag([2.0, 1.0, 0.3])
B2 = Uq @ np.sqrt(D) @ Q.T; B1 = Q @ np.sqrt(D) @ Vq.T
assert np.allclose(B1 @ B1.T, B2.T @ B2)
Wb = B2 @ B1
assert np.allclose(B2 @ B2.T, np.real(sqrtm(Wb @ Wb.T))) and np.allclose(B1.T @ B1, np.real(sqrtm(Wb.T @ Wb)))
ok("balanced: W2W2^T = (WW^T)^1/2, W1^TW1 = (W^TW)^1/2")
# lambda-balanced, square case: W1W1^T - W2^TW2 = lam I  (h = nx = ny)
m, lam = 4, 0.7
Ws = rng.normal(size=(m, m)); Uw, sw, Vtw = np.linalg.svd(Ws)
# per singular value: b^2 - c^2 = lam, bc = sw  => b^2 = (lam + sqrt(lam^2+4 sw^2))/2
bb = np.sqrt((lam + np.sqrt(lam**2 + 4 * sw**2)) / 2); cc = sw / bb
Qm, _ = np.linalg.qr(rng.normal(size=(m, m)))
A1 = Qm @ np.diag(bb) @ Vtw; A2 = Uw @ np.diag(cc) @ Qm.T
assert np.allclose(A1 @ A1.T - A2.T @ A2, lam * np.eye(m)) and np.allclose(A2 @ A1, Ws)
Wm = A2 @ A1
P = np.real(sqrtm(Wm @ Wm.T + lam**2 / 4 * np.eye(m))); Pt = np.real(sqrtm(Wm.T @ Wm + lam**2 / 4 * np.eye(m)))
assert np.allclose(A2 @ A2.T, P - lam / 2 * np.eye(m)) and np.allclose(A1.T @ A1, Pt + lam / 2 * np.eye(m))
ok("lambda-balanced (C=W1W1^T-W2^TW2=lam I): W2W2^T = sqrt(WW^T+lam^2/4) - lam/2, W1^TW1 = sqrt(W^TW+lam^2/4) + lam/2")

# ---------------------------------------------------------------- 8. linearized small-weight dynamics: Z' = Z M
Ss = rng.normal(size=(ny, nx))
M = np.block([[np.zeros((nx, nx)), Ss.T], [Ss, np.zeros((ny, ny))]])
Z1, Z2 = rng.normal(size=(h, nx)), rng.normal(size=(h, ny))  # Z1 = W1, Z2 = W2^T
Z = np.hstack([Z1, Z2])
lin1, lin2 = Z2 @ Ss, Z1 @ Ss.T  # W2^T S, (S W1^T)^T = W1 S^T
assert np.allclose(Z @ M, np.hstack([lin1, lin2]))
Uu, ss, Vt = np.linalg.svd(Ss)
ev = np.linalg.eigvalsh(M)
expected = np.sort(np.concatenate([ss, -ss, np.zeros(nx + ny - 2 * len(ss))]))
assert np.allclose(np.sort(ev), expected)
q = np.concatenate([Vt[0], Uu[:, 0]]) / np.sqrt(2)
assert np.allclose(M @ q, ss[0] * q); ok("Z' = Z M, eig(M) = +-s_a (+zeros), eigvec (v_a, +-u_a)/sqrt2")
# conserved imbalance also conserved by linearized flow: MJ + JM = 0
Jm = np.diag(np.concatenate([np.ones(nx), -np.ones(ny)]))
assert np.allclose(M @ Jm + Jm @ M, 0); ok("linearized flow conserves Z J Z^T = W1W1^T - W2^TW2")
print("all checks passed")

# ---------------------------------------------------------------- 9. per-mode invariant Delta (Sec. 4.3)
pv, qv = sp.symbols('p1:4', real=True), sp.symbols('q1:4', real=True)
P = sp.Matrix(pv); Qm = sp.Matrix(qv); sv_ = sp.symbols('s', positive=True)
am = (P.T * Qm)[0]; km = (P.T * P)[0] + (Qm.T * Qm)[0]
pdot = Qm * (sv_ - am); qdot = P * (sv_ - am)            # tau p' = q(s-a), tau q' = p(s-a)
adot = (pdot.T * Qm + P.T * qdot)[0]; kdot = (2 * P.T * pdot + 2 * Qm.T * qdot)[0]
assert sp.simplify(adot - km * (sv_ - am)) == 0 and sp.simplify(kdot - 4 * am * (sv_ - am)) == 0
assert sp.simplify(2 * km * kdot - 8 * am * adot) == 0
ok("tau a' = k(s-a), tau k' = 4a(s-a), so k^2-4a^2 is conserved")
for _ in range(5):
    p_, q_ = rng.normal(size=5), rng.normal(size=5)
    D2 = (q_ @ q_ - p_ @ p_)**2 + 4 * ((p_ @ p_) * (q_ @ q_) - (p_ @ q_)**2)
    nuc = np.sum(np.abs(np.linalg.eigvalsh(np.outer(q_, q_) - np.outer(p_, p_))))
    kk_, aa_ = p_ @ p_ + q_ @ q_, p_ @ q_
    assert np.isclose(np.sqrt(D2), nuc) and np.isclose(D2, kk_**2 - 4 * aa_**2)
ok("Delta^2 = (|q|^2-|p|^2)^2 + 4|p^q|^2 = ||qq^T-pp^T||_*^2 = k^2-4a^2")
# effective a0 = Delta/4 when s >> Delta, a0 = 0
for sv, Dv in [(1.0, 1e-3), (5.0, 0.1), (2.0, 1e-4)]:
    R_ = np.sqrt(4 * sv**2 + Dv**2)
    F_ = lambda x: np.log((4 * sv * x + Dv**2 + R_ * np.sqrt(4 * x**2 + Dv**2)) / (2 * sv - 2 * x)) / R_
    th_exact = F_(sv / 2) - F_(0.0)
    th_bal = np.log(sv / (Dv / 4) - 1) / (2 * sv)
    assert abs(th_exact / th_bal - 1) < 2e-3, (sv, Dv, th_exact, th_bal)
ok("a0 -> 0 with s >> Delta: half-time = balanced half-time with a0 = Delta/4")
# wide Gaussian: Delta ~ 2 sigma^2 h ; fan-in: Delta ~ h/nx + 1
h_, nx_, ny_ = 4000, 6, 6
Vv, _ = np.linalg.qr(rng.normal(size=(nx_, nx_))); Uu_, _ = np.linalg.qr(rng.normal(size=(ny_, ny_)))
for s1, s2, pred in [(0.01, 0.01, 2 * 0.01 * h_), (1 / nx_, 1 / h_, h_ / nx_ + 1)]:
    A1 = np.sqrt(s1) * rng.normal(size=(h_, nx_)); A2 = np.sqrt(s2) * rng.normal(size=(ny_, h_))
    p_, q_ = A1 @ Vv[:, 0], A2.T @ Uu_[:, 0]
    Dl_ = np.sqrt((p_ @ p_ + q_ @ q_)**2 - 4 * (p_ @ q_)**2)
    assert abs(Dl_ / pred - 1) < 0.1, (Dl_, pred)
ok("Gaussian init: Delta ~ 2 sigma^2 h (equal variances), ~ h/nx + 1 (fan-in)")
# Ex 2.3: Hessian of 1/2(s - prod w)^2 at balanced minimum has top eigenvalue N s^(2-2/N)
for Nv in [2, 3, 5]:
    ws = sp.symbols(f'w1:{Nv + 1}', positive=True)
    Lw = sp.Rational(1, 2) * (s - sp.Mul(*ws))**2
    Hm = sp.hessian(Lw, ws).subs({x: s**sp.Rational(1, Nv) for x in ws})
    sval = 2.3
    ev_ = np.linalg.eigvalsh(np.array(Hm.subs(s, sval), dtype=float))
    assert np.isclose(ev_.max(), Nv * sval**(2 - 2 / Nv)) and np.sum(np.abs(ev_) < 1e-9) == Nv - 1
ok("depth-N balanced minimum: Hessian eigenvalues {N s^(2-2/N), 0 x (N-1)}")
# Ex 6.1 numbers
t1 = 0.5 * np.log(999); t2 = np.log(899) / 1.8; t2b = np.log(499)
assert abs(t1 - 3.45) < 0.01 and abs(t2 - 3.78) < 0.01 and abs(t2b - 6.21) < 0.01 and abs(np.log(9) - 2.197) < 1e-3
ok("Exercise 6.1 half-times 3.45, 3.78, 6.21 tau; 10-90% width ln9 tau/s")
print("all checks passed (extended)")

# ---------------------------------------------------------------- 10. round-3 additions
# (a) at fixed end-to-end a0 the N>=3 plateau peaks near N = 2 ln(1/a0) - 2
for a0v in [1e-3, 1e-6]:
    Ns = np.arange(3, 80)
    te = [quad(lambda x: 1 / (Nv * x**(2 - 2 / Nv) * (1 - x)), a0v, 0.5, limit=500)[0] for Nv in Ns]
    assert abs(Ns[int(np.argmax(te))] - (2 * np.log(1 / a0v) - 2)) <= 1.5
ok("fixed end-to-end a0: plateau peaks at N ~ 2 ln(1/a0) - 2")
# (b) balanced square flow: d/dt ln|det W| = 2 tr(O^T E)/tau, O the polar factor
for _ in range(3):
    A_ = rng.normal(size=(4, 4)); Ua_, sa_, Vta_ = np.linalg.svd(A_); Qo = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    B1 = Qo @ np.diag(np.sqrt(sa_)) @ Vta_; B2 = Ua_ @ np.diag(np.sqrt(sa_)) @ Qo.T
    Sx = rng.normal(size=(4, 4)); Wx = B2 @ B1; Ex = Sx - Wx
    Wdot = B2 @ B2.T @ Ex + Ex @ B1.T @ B1
    Uw, _, Vtw = np.linalg.svd(Wx)
    assert np.isclose(np.trace(np.linalg.solve(Wx, Wdot)), 2 * np.trace((Uw @ Vtw).T @ Ex))
ok("balanced: tau d/dt ln|det W| = 2 tr(O^T E)")
# (c) S-shaped rise iff s > sqrt(2) Delta : f'(a) > 0 for some a in (0,s)
for sv, Dv in [(1.0, 0.6), (1.0, 0.8), (3.0, 2.0), (2.5, 2.0)]:
    aa = np.linspace(0, sv, 20001)
    fp = (4 * aa * sv - 8 * aa**2 - Dv**2) / np.sqrt(4 * aa**2 + Dv**2)
    assert (fp.max() > 0) == (sv > np.sqrt(2) * Dv), (sv, Dv)
ok("convex (S-shaped) rise exists iff s > sqrt(2) Delta")
# (d) alignment ratio a2/a1 at t1 (factor 2): decoupled logistics, equal tiny seeds
s1, s2, a0v = 3.0, 2.0, 1e-12
t1 = np.log(s1 / a0v - 1) / (2 * s1)
lg = lambda t, s_, a_: s_ * np.exp(2 * s_ * t) / (np.exp(2 * s_ * t) - 1 + s_ / a_)
ratio = lg(t1, s2, a0v) / lg(t1, s1, a0v); pred = 2 * (a0v / s1)**(1 - s2 / s1)
assert abs(ratio / pred - 1) < 1e-3
ok("a2/a1 at mode-1 half-time = 2 (a02/a01)(a01/s1)^(1-s2/s1)")
# (e) balanced scalar Hessian at a=15, s=3 has eigenvalues 42, -12 (kernel 30)
bb_ = np.sqrt(15.0); Hs = np.array([[bb_**2, 2 * bb_**2 - 3], [2 * bb_**2 - 3, bb_**2]])
assert np.allclose(np.sort(np.linalg.eigvalsh(Hs)), [-12, 42])
ok("balanced scalar mode a=15, s=3: Hessian eigenvalues {42, -12}")
print("all checks passed (round 3)")
