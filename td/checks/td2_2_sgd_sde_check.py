#!/usr/bin/env python3
"""Checks for every closed-form result stated in td2_2_sgd_sde.

Run: python td/checks/td2_2_sgd_sde_check.py   (about 20 s on a laptop)
Each block prints OK or raises.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(0)


def ok(msg):
    print("OK  ", msg)


# ---------------------------------------------------------------- 1. SME moments
# Generator expansion: E phi(X_eta) = phi + eta*Lphi + eta^2/2 L^2 phi + O(eta^3)
# for dX = b dt + sqrt(D) dW.  Take b = -grad f + eta*b1, D = eta*s (s = Sigma/B).
def sme_moments_2d():
    x1, x2, y1, y2, eta = sp.symbols("x1 x2 y1 y2 eta", real=True)
    Y = sp.Matrix([y1, y2])
    f = y1**4 / 4 + y1**2 * y2 + 3 * y2**2 / 2 + y1 * y2**3 / 5  # a generic polynomial loss
    G = sp.Matrix([sp.diff(f, v) for v in (y1, y2)])
    H = G.jacobian(Y)
    b1 = -sp.Rational(1, 2) * H * G  # claimed drift correction  (= -1/4 grad|G|^2)
    assert sp.simplify(b1 + sp.Rational(1, 4) * sp.Matrix([sp.diff(G.dot(G), v) for v in (y1, y2)])) == sp.zeros(2, 1)
    s11, s12, s22 = sp.symbols("s11 s12 s22")
    S = sp.Matrix([[s11 + y1**2, s12], [s12, s22 + y2**2]])  # state-dependent noise covariance
    b = -G + eta * b1
    D = eta * S

    def gen(phi):
        grad = sp.Matrix([sp.diff(phi, v) for v in (y1, y2)])
        hess = grad.jacobian(Y)
        return (b.T * grad)[0] + sp.Rational(1, 2) * sum(D[i, j] * hess[i, j] for i in range(2) for j in range(2))

    def moment(phi):
        val = phi + eta * gen(phi) + eta**2 / 2 * gen(gen(phi))
        val = val.subs({y1: x1, y2: x2})
        return sp.expand(sp.series(sp.expand(val), eta, 0, 3).removeO())

    sub = {y1: x1, y2: x2}
    Gx, Sx = G.subs(sub), S.subs(sub)
    # first moments: SGD gives exactly -eta*G
    for i, yi in enumerate((y1, y2)):
        m = moment(yi - [x1, x2][i])
        assert sp.simplify(m - (-eta * Gx[i])) == 0, m
    # second moments: SGD gives eta^2 (G G^T + S)
    for i in range(2):
        for j in range(2):
            phi = (Y[i] - [x1, x2][i]) * (Y[j] - [x1, x2][j])
            m = moment(phi)
            assert sp.simplify(m - eta**2 * (Gx[i] * Gx[j] + Sx[i, j])) == 0
    # without the correction the first moment is off at eta^2 by +eta^2/2 H G
    b_plain = -G
    def gen0(phi):
        grad = sp.Matrix([sp.diff(phi, v) for v in (y1, y2)])
        hess = grad.jacobian(Y)
        return (b_plain.T * grad)[0] + sp.Rational(1, 2) * sum(D[i, j] * hess[i, j] for i in range(2) for j in range(2))
    m0 = (y1 - x1) + eta * gen0(y1 - x1) + eta**2 / 2 * gen0(gen0(y1 - x1))
    m0 = sp.expand(sp.series(sp.expand(m0.subs(sub)), eta, 0, 3).removeO())
    err = sp.simplify(m0 - (-eta * Gx[0]))
    assert sp.simplify(err - eta**2 / 2 * (H * G).subs(sub)[0]) == 0
    ok("order-2 SME: drift correction -1/2 H G = -1/4 grad|G|^2 matches SGD's first two moments to O(eta^3); "
       "without it the mean is off by +eta^2/2 HG")


sme_moments_2d()

# ---------------------------------------------------------------- 2. Fokker-Planck / OU
d = 4
A = rng.standard_normal((d, d)); H = A @ A.T + 0.5 * np.eye(d)
Bm = rng.standard_normal((d, d)); D = Bm @ Bm.T + 0.1 * np.eye(d)


def lyap(H, D):
    """Solve H P + P H = D in H's eigenbasis."""
    h, U = np.linalg.eigh(H)
    Dt = U.T @ D @ U
    return U @ (Dt / (h[:, None] + h[None, :])) @ U.T


P = lyap(H, D)
assert np.allclose(H @ P + P @ H, D)
# Gaussian N(0,P) is stationary for dtheta = -H theta dt + D^{1/2} dW iff FP residual vanishes.
# FP: dp/dt = div(H theta p) + 1/2 sum D_ij d_i d_j p.  For p = N(0,P): grad p = -P^{-1} theta p,
# Hess p = (P^{-1} theta theta^T P^{-1} - P^{-1}) p.  Residual/p is a quadratic form in theta plus const.
Pi = np.linalg.inv(P)
const = np.trace(H) - 0.5 * np.trace(D @ Pi)
quad = -(H.T @ Pi) + 0.5 * Pi @ D @ Pi  # theta^T quad theta
quad_sym = (quad + quad.T) / 2
assert abs(const) < 1e-10 and np.abs(quad_sym).max() < 1e-10
ok("OU: Gaussian with Lyapunov covariance makes the Fokker-Planck right-hand side vanish")
# Boltzmann e^{-c L} has covariance H^{-1}/c; Lyapunov then forces D = (2/c) I.
c = 3.0
Pb = np.linalg.inv(H) / c
assert np.allclose(H @ Pb + Pb @ H, 2 / c * np.eye(d))
# D proportional to H gives isotropic P = (k/2) I, not proportional to H^{-1}
k = 0.7
assert np.allclose(lyap(H, k * H), k / 2 * np.eye(d))
ok("Boltzmann (P ~ H^-1) iff D ~ I; D = k H gives P = k/2 I (isotropic, not Boltzmann)")
# general potential, constant isotropic D = 2T I: e^{-L/T} has zero probability current (symbolic, 2D)
x, y, T = sp.symbols("x y T", positive=True)
Lxy = x**4 + x * y + y**2 * sp.cos(x)
p = sp.exp(-Lxy / T)
J = [-sp.diff(Lxy, v) * p - T * sp.diff(p, v) for v in (x, y)]
assert all(sp.simplify(j) == 0 for j in J)
ok("isotropic constant noise: Gibbs density has zero current for a non-quadratic potential")
# trace identities
eta, B = 0.05, 8
Sig = D
Pc = lyap(H, eta / B * Sig)
assert np.isclose(0.5 * np.trace(H @ Pc), eta / (4 * B) * np.trace(Sig))
assert np.isclose(np.trace(H @ H @ Pc), eta / (2 * B) * np.trace(H @ Sig))
ok("excess loss 1/2 tr(H Pi) = eta trSigma/4B and E|G|^2 = tr(H^2 Pi) = eta tr(H Sigma)/2B, any Sigma")

# ---------------------------------------------------------------- 3. discrete stationary covariance
# additive noise: Pi = (I - eta H) Pi (I - eta H) + eta^2 Sigma/B  <=>  H Pi + Pi H - eta H Pi H = eta Sigma/B
def disc_additive(H, Sig, eta, B):
    dd = H.shape[0]
    M = np.eye(dd) - eta * H
    K = np.eye(dd * dd) - np.kron(M, M)
    return np.linalg.solve(K, (eta**2 / B * Sig).reshape(-1)).reshape(dd, dd)


lam_max = np.linalg.eigvalsh(H).max()
for frac in (0.05, 0.5, 0.95):                    # eta * lambda_max < 2: a stationary covariance exists
    et = 2 * frac / lam_max
    Pd = disc_additive(H, Sig, et, B)
    assert np.allclose(H @ Pd + Pd @ H - et * H @ Pd @ H, et / B * Sig)
    assert np.linalg.eigvalsh(Pd).min() > 0       # a genuine covariance
# beyond the edge the linear equation still has a solution, but it is not a covariance
Pbad = disc_additive(H, Sig, 2.5 / lam_max, B)
assert np.linalg.eigvalsh(Pbad).min() < 0
# commuting case: excess loss = eta/(2B) sum_j Sigma_jj / (2 - eta h_j)
h = np.array([0.5, 1.0, 3.0, 6.0]); Sd = np.diag([1.0, 2.0, 0.5, 1.5])
et = 0.25
Pd = disc_additive(np.diag(h), Sd, et, B)
assert np.isclose(0.5 * np.trace(np.diag(h) @ Pd), et / (2 * B) * np.sum(np.diag(Sd) / (2 - et * h)))
ok("discrete stationary covariance and the 1/(1 - eta h/2) enhancement (commuting case)")

# ---------------------------------------------------------------- 4. McCandlish algebra
e, g2, gHg, trHS, Bs = sp.symbols("epsilon g2 gHg trHS B", positive=True)
EL = -e * g2 + e**2 / 2 * (gHg + trHS / Bs)
eopt = sp.solve(sp.diff(EL, e), e)[0]
emax = g2 / gHg
Bnoise = trHS / gHg
assert sp.simplify(eopt - emax / (1 + Bnoise / Bs)) == 0
dLopt = -EL.subs(e, eopt)
dLmax = g2**2 / (2 * gHg)
assert sp.simplify(dLopt - dLmax / (1 + Bnoise / Bs)) == 0
Smin, Bn, Bv = sp.symbols("S_min B_noise Bv", positive=True)
S_ = Smin * (1 + Bn / Bv); E_ = S_ * Bv; Emin = Smin * Bn
assert sp.simplify((S_ / Smin - 1) * (E_ / Emin - 1) - 1) == 0
assert sp.simplify(Emin / Smin - Bn) == 0
ok("eps_opt, dL_opt, (S/Smin-1)(E/Emin-1)=1, Bcrit = Bnoise")

# ---------------------------------------------------------------- 5. two-batch estimator (with replacement)
N, dd = 400, 6
grads = rng.standard_normal((N, dd)) * np.array([3, 1, 1, .5, .5, .2]) + np.array([0.4, 0.1, 0, 0, 0, 0])
G = grads.mean(0); trS = ((grads - G) ** 2).sum(1).mean()
bs, bb, R = 4, 64, 100000
idx_s = rng.integers(0, N, (R, bs)); idx_b = rng.integers(0, N, (R, bb))
ns = (grads[idx_s].mean(1) ** 2).sum(1); nb = (grads[idx_b].mean(1) ** 2).sum(1)
G2_est = (bb * nb - bs * ns) / (bb - bs); S_est = (ns - nb) / (1 / bs - 1 / bb)
assert abs(G2_est.mean() - G @ G) < 0.02 * (G @ G) + 4 * G2_est.std() / np.sqrt(R)
assert abs(S_est.mean() - trS) < 4 * S_est.std() / np.sqrt(R)
# without replacement: E|g_B|^2 = |G|^2 + (N-B)/(N-1) trSigma/B
perm = np.argsort(rng.random((20000, N)), axis=1)[:, :bb]
nwo = (grads[perm].mean(1) ** 2).sum(1)
pred = G @ G + (N - bb) / (N - 1) * trS / bb
assert abs(nwo.mean() - pred) < 4 * nwo.std() / np.sqrt(len(nwo))
ok("two-batch estimators unbiased (with replacement); without-replacement factor (N-B)/(N-1)")

# ---------------------------------------------------------------- 6. weak order on a 1D quadratic (exact moments)
def weak_errors(etas, h=1.0, s2=1.0, Bb=1, x0=1.0, Tt=2.0):
    out = []
    for et in etas:
        kk = int(round(Tt / et))
        t = kk * et
        q = 1 - et * h
        m_sgd = q**kk * x0
        v_sgd = et**2 * s2 / Bb * (1 - q ** (2 * kk)) / (1 - q**2)
        e_sgd = m_sgd**2 + v_sgd
        e_gf = (np.exp(-h * t) * x0) ** 2
        a1 = h
        e_1 = (np.exp(-a1 * t) * x0) ** 2 + et * s2 / Bb * (1 - np.exp(-2 * a1 * t)) / (2 * a1)
        a2 = h + et * h**2 / 2
        e_2 = (np.exp(-a2 * t) * x0) ** 2 + et * s2 / Bb * (1 - np.exp(-2 * a2 * t)) / (2 * a2)
        out.append([abs(e_gf - e_sgd), abs(e_1 - e_sgd), abs(e_2 - e_sgd)])
    return np.array(out)


etas = np.array([0.02, 0.01, 0.005, 0.0025])
for x0, expect in ((3.0, (1, 1, 2)), (0.0, (1, 2, 2))):
    errs = weak_errors(etas, x0=x0)
    slopes = np.polyfit(np.log(etas), np.log(errs), 1)[0]
    assert np.allclose(slopes, expect, atol=0.1), (x0, slopes)
ok("weak order on a quadratic: GF/SME1/SME2 slopes (1,1,2) from x0=3 and (1,2,2) from x0=0")

# ---------------------------------------------------------------- 7. square-root heuristic for RMSprop
# noise-dominated: v ~ diag(Sigma)/B, so per-step drift ~ eta sqrt(B) and per-step noise var ~ eta^2:
# temperature eta/sqrt(B), time per step eta sqrt(B).  Check that eta -> sqrt(k) eta keeps eta/sqrt(B).
kap = sp.symbols("kappa", positive=True)
et_, B_ = sp.symbols("eta B", positive=True)
assert sp.simplify((sp.sqrt(kap) * et_) / sp.sqrt(kap * B_) - et_ / sp.sqrt(B_)) == 0
assert sp.simplify((sp.sqrt(kap) * et_) * sp.sqrt(kap * B_) - kap * et_ * sp.sqrt(B_)) == 0
ok("square-root rule keeps eta/sqrt(B) fixed and multiplies time per step by kappa")
print("all checks passed")

# ---------------------------------------------------------------- 8. more claims in the text
# (a) order-2 SME OU stationary variance has the opposite-sign O(eta h) correction to discrete SGD
hh, et_v, sv, Bv_ = 1.3, 0.1, 2.0, 4
sgd_var = et_v * sv / (Bv_ * hh * (2 - et_v * hh))
sme2_var = et_v * sv / (Bv_ * hh * (2 + et_v * hh))
ou_var = et_v * sv / (Bv_ * 2 * hh)
assert sme2_var < ou_var < sgd_var
ok("order-2 SME stationary variance correction has the wrong sign (narrower, truth is wider)")
# (b) preconditioned floor 1/2 tr(H Pi) = eta tr(P Sigma)/4B for a non-commuting P
Pm = rng.standard_normal((d, d)); Pm = Pm @ Pm.T + np.eye(d)
A_ = Pm @ H                     # drift matrix
# solve A Pi + Pi A^T = eta/B P Sig P via vectorization
I_ = np.eye(d)
K = np.kron(A_, I_) + np.kron(I_, A_)
rhs = (eta / B * Pm @ Sig @ Pm).reshape(-1)
Pp = np.linalg.solve(K, rhs).reshape(d, d)
assert np.allclose(A_ @ Pp + Pp @ A_.T, eta / B * Pm @ Sig @ Pm)
assert np.isclose(0.5 * np.trace(H @ Pp), eta / (4 * B) * np.trace(Pm @ Sig))
ok("preconditioned OU floor: 1/2 tr(H Pi) = eta tr(P Sigma)/4B for general P")
# (c) SVAG coefficients
for l in (1, 2, 5, 32):
    s_ = np.sqrt(2 * l - 1); ap, am = (1 + s_) / 2, (1 - s_) / 2
    assert np.isclose(ap + am, 1) and np.isclose(ap**2 + am**2, l)
ok("SVAG: a+ + a- = 1, a+^2 + a-^2 = l")
# (d) 10% threshold: (1 - x/2)^-1 <= 1.1  <=>  x <= 2/11
assert np.isclose(2 / 11, 0.1818, atol=1e-4)
# (e) equilibrium B_simple = 2B trS/(eta tr HS) from the trace identities
Pc = lyap(H, eta / B * Sig)
assert np.isclose(np.trace(Sig) / np.trace(H @ H @ Pc), 2 * B * np.trace(Sig) / (eta * np.trace(H @ Sig)))
ok("equilibrium B_simple = 2B trSigma / (eta tr(H Sigma))")
print("all checks passed (extended)")

# (f) exponent family: P = (diag Sigma / B)^(-p) gives floor eta B^(p-1)/4 sum Sigma_jj^(1-p)
for pexp in (0.0, 0.5, 1.0):
    for Bq in (2, 16):
        Pq = np.diag((np.diag(Sig) / Bq) ** (-pexp))
        lhs = eta / (4 * Bq) * np.trace(Pq @ Sig)
        assert np.isclose(lhs, eta * Bq ** (pexp - 1) / 4 * np.sum(np.diag(Sig) ** (1 - pexp)))
ok("exponent family: floor depends on eta * B^(p-1)")
# (g) one discrete mode: N/G = (2 - eta h)/(eta h); LMA threshold with C^2 = 2 is kappa*eta*h > 4
for eh in (0.1, 1.0, 1.5):
    Pi1 = eh / hh * sv / (Bv_ * hh * (2 - eh))     # eta = eh/hh
    NG = (sv / Bv_) / (hh**2 * Pi1)
    assert np.isclose(NG, (2 - eh) / eh)
    assert np.isclose(2 * (1 + NG) * eh, 4)
ok("discrete one-mode N/G = (2 - eta h)/(eta h); criterion kappa > C^2(1+N/G) <=> kappa eta h > 4 at C^2=2")
# (h) SVAG(l) one mode: floor = (eta Sigma / 4B) (1 - eta h / (2l))^-1
for l in (1, 2, 4, 8):
    eta_, B_, s_, h_ = 0.9, 8, 1.7, 2.0
    lr = eta_ / l
    Pi_ = (lr**2 * l * s_ / B_) / (1 - (1 - lr * h_) ** 2)
    assert np.isclose(0.5 * h_ * Pi_, eta_ * s_ / (4 * B_) / (1 - eta_ * h_ / (2 * l)))
ok("SVAG one-mode floor (eta Sigma/4B)(1 - eta h/2l)^-1")
