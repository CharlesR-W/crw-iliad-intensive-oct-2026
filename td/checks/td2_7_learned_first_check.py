#!/usr/bin/env python3
"""Checks for the closed-form statements in td2_7_learned_first (NB2 §7).

Run: python td/checks/td2_7_learned_first_check.py   (about 20 s on a laptop)
Each check prints its numbers and asserts agreement.
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import ive
from scipy.stats import norm

rng = np.random.default_rng(0)
ok = lambda name: print(f"[ok] {name}")

# ---------------------------------------------------------------- 7.1.B: GELU Taylor coefficients
z = sp.symbols("z")
gelu = z * (1 + sp.erf(z / sp.sqrt(2))) / 2
ser = sp.series(gelu, z, 0, 6).removeO()
c = [sp.nsimplify(ser.coeff(z, k)) for k in range(6)]
print("GELU coefficients c0..c5:", c)
assert c[1] == sp.Rational(1, 2) and sp.simplify(c[2] - 1 / sp.sqrt(2 * sp.pi)) == 0 and c[3] == 0
assert sp.simplify(c[4] + 1 / (6 * sp.sqrt(2 * sp.pi))) == 0
ok("GELU: c1=1/2, c2=1/sqrt(2 pi), c3=0, c4=-1/(6 sqrt(2 pi))")

# ---------------------------------------------------------------- 7.1.B.4: data vs clone gradient gap ~ eps^4
# Exact expectations by Gauss-Hermite quadrature in the 3 spike coordinates (m, u, v);
# the other d-3 coordinates are N(0,1) in both classes and are set to zero weight here.
bm, bu, bv = 1.0, 5.0, 10.0
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40)
gh_w = gh_w / gh_w.sum()
G1, G2, G3 = np.meshgrid(gh_x, gh_x, gh_x, indexing="ij")
Wq = (gh_w[:, None, None] * gh_w[None, :, None] * gh_w[None, None, :]).ravel()
Zq = np.stack([G1.ravel(), G2.ravel(), G3.ravel()], 1)
mu_v, s_v = np.sqrt(bv / (1 + bv)), 1 / np.sqrt(1 + bv)
neg = (Zq, Wq)
def plus(clone):
    base = np.c_[bm + Zq[:, 0], np.sqrt(1 + bu) * Zq[:, 1], Zq[:, 2]]
    if clone:
        return [(base, Wq)]
    a = base.copy(); a[:, 2] = mu_v + s_v * Zq[:, 2]
    b = base.copy(); b[:, 2] = -mu_v + s_v * Zq[:, 2]
    return [(a, Wq / 2), (b, Wq / 2)]

import torch
torch.set_default_dtype(torch.float64)
gl = torch.nn.functional.gelu
torch.manual_seed(0)
W0, a0 = torch.randn(8, 3), torch.randn(8)
def grad(eps, clone):
    W = (eps * W0).requires_grad_(); a = (eps * a0).requires_grad_()
    L = 0
    for (X, w), y in [(neg, -1.0)] + [(p, 1.0) for p in plus(clone)]:
        X, w = torch.tensor(X), torch.tensor(w)
        L = L + 0.25 * (w * (y - gl(X @ W.T) @ a) ** 2).sum()
    # weights: class -1 has prob 1/2, class +1 components already split; 0.25 = (1/2)*(1/2 for the square)
    gW, ga = torch.autograd.grad(L, (W, a))
    return torch.cat([gW.ravel(), ga])
epss = np.array([0.004, 0.008, 0.016, 0.032])
gaps = [float((grad(e, False) - grad(e, True)).norm()) for e in epss]
lead = [float(grad(e, False).norm()) for e in epss]
sl_gap = np.polyfit(np.log(epss), np.log(gaps), 1)[0]
sl_lead = np.polyfit(np.log(epss), np.log(lead), 1)[0]
print(f"|grad_data - grad_clone| slope in eps: {sl_gap:.2f} (claim 4);  |grad| slope: {sl_lead:.2f} (claim 1)")
assert abs(sl_gap - 4) < 0.15 and abs(sl_lead - 1) < 0.1
ok("7.1.B.4: data/clone gradients differ first at O(eps^4) for GELU on the mixed cumulant model")

# odd activation (tanh): first difference through E[x^{(4)}] at eps^5
def grad_t(eps, clone):
    W = (eps * W0).requires_grad_(); a = (eps * a0).requires_grad_()
    L = 0
    for (X, w), y in [(neg, -1.0)] + [(p, 1.0) for p in plus(clone)]:
        X, w = torch.tensor(X), torch.tensor(w)
        L = L + 0.25 * (w * (y - torch.tanh(X @ W.T) @ a) ** 2).sum()
    gW, ga = torch.autograd.grad(L, (W, a))
    return torch.cat([gW.ravel(), ga])
gaps_t = [float((grad_t(e, False) - grad_t(e, True)).norm()) for e in epss]
sl_t = np.polyfit(np.log(epss), np.log(gaps_t), 1)[0]
print(f"tanh: |grad_data - grad_clone| slope {sl_t:.2f} (claim 5)")
assert abs(sl_t - 5) < 0.15
ok("7.1.B.4: for an odd activation the first difference is O(eps^5)")
torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------- 7.1.C: tanh numbers
g = quad(lambda x: np.tanh(x) ** 2 * norm.pdf(x), -np.inf, np.inf)[0]
print(f"E tanh^2(x), Gaussian: {g:.4f}; Rademacher: {np.tanh(1)**2:.4f}")
assert abs(g - 0.3943) < 5e-5 and abs(np.tanh(1) ** 2 - 0.5800) < 5e-5
u_ = sp.symbols("u")
print("tanh^2 series:", sp.series(sp.tanh(u_) ** 2, u_, 0, 8))
for w in (0.1, 0.05):
    G = quad(lambda x: np.tanh(w * x) ** 2 * norm.pdf(x), -np.inf, np.inf)[0]
    exact, approx = G - np.tanh(w) ** 2, -(2 / 3) * w**4 * 2 + (17 / 45) * w**6 * 14
    print(f"w={w}: exact difference {exact:.4e}, series {approx:.4e}")
    assert abs(exact - approx) < 3 * w**8 * 100
ok("7.1.C: 0.3943 vs 0.5800, first difference -(4/3) w^4")

# ---------------------------------------------------------------- 7.3.A.2: closed form vs integrating the flow
n = 60
th = rng.uniform(0, 2 * np.pi, n)
K = np.exp(4 * (np.cos(th[:, None] - th[None, :]) - 1))
y = np.sin(th) + np.sin(5 * th)
lam, V = np.linalg.eigh(K / n)
t_end, dt = 50.0, 1e-3
alpha = np.zeros(n)
for _ in range(int(t_end / dt)):  # theta-flow in the span: d alpha/dt = -(1/n)(K alpha - y)
    alpha -= dt * (K @ alpha - y) / n
x0 = np.array([0.3, 2.0])
Kx = np.exp(4 * (np.cos(x0[:, None] - th[None, :]) - 1))
f_num = Kx @ alpha
filt = -np.expm1(-np.clip(lam, 0, None) * t_end) / np.maximum(lam, 1e-14)
f_cf = Kx @ (V @ (filt * (V.T @ y))) / n
print("f_t(x) by integration vs closed form:", f_num, f_cf)
assert np.allclose(f_num, f_cf, rtol=1e-3, atol=1e-3)
ok("7.3.A.2: f_t(x) closed form")

# ---------------------------------------------------------------- 7.3.B: von Mises eigenvalues and asymptotics
kap, N = 4.0, 1024
grid = 2 * np.pi * np.arange(N) / N
ev = np.fft.rfft(np.exp(kap * (np.cos(grid) - 1))).real / N
assert np.allclose(ev[:20], ive(np.arange(20), kap), atol=1e-15)
print("k_1, k_4, k_8 =", ive([1, 4, 8], kap), " t8/t1 =", ive(1, kap) / ive(8, kap))
assert abs(ive(1, kap) / ive(8, kap) - 995) < 5
m = np.arange(12, 30)
ratio = ive(m, kap) / (np.exp(-kap) * (kap / 2) ** m / np.array([float(sp.factorial(int(k))) for k in m]))
print("small-kernel-limit ratio, m=12..29:", np.round(ratio[[0, 6, 17]], 3))
assert abs(ratio[-1] - 1) < 0.15
kap2 = 400.0
mm = np.arange(0, 15)
print("large kappa: e^{-kappa}I_m / [e^{-m^2/2k}/sqrt(2 pi k)] =",
      np.round(ive(mm, kap2) / (np.exp(-mm**2 / (2 * kap2)) / np.sqrt(2 * np.pi * kap2)), 4)[[0, 7, 14]])
assert np.allclose(ive(mm, kap2) / (np.exp(-mm**2 / (2 * kap2)) / np.sqrt(2 * np.pi * kap2)), 1, atol=0.01)
ok("7.3.B: e^{-k} I_m(k), asymptotics, t8/t1 ~ 1000")

# ---------------------------------------------------------------- 7.3.C: Basri et al. eq. (12)
Knb = lambda t: np.cos(t) * (np.pi - abs(t)) / (2 * np.pi)
def coef(F, k):
    return quad(lambda t: F(t) * np.cos(k * t), -np.pi, np.pi, points=[0], limit=200)[0] / (2 * np.pi if k == 0 else np.pi)
def basri(k):
    return 1 / np.pi**2 if k == 0 else 0.25 if k == 1 else (2 * (k * k + 1) / (np.pi**2 * (k * k - 1) ** 2) if k % 2 == 0 else 0)
assert max(abs(coef(Knb, k) - basri(k)) for k in range(16)) < 1e-12
ok("7.3.C: Basri et al. eq. (12) by quadrature")
def Kb_(t):
    u = (1 + np.cos(t)) / 2
    return u * (np.pi - np.arccos(np.clip(u, -1, 1))) / (2 * np.pi)
r18 = coef(Kb_, 1) / coef(Kb_, 8)
print(f"bias kernel a_1/a_8 = {r18:.1f}"); assert abs(r18 - 200) < 5
ok("7.3.B.3: a_1/a_8 ~ 200 for the bias kernel")
# 7.1.B.4: E[y x_m x_v^4] on data vs clone (beta_m = 1): (3 + k4)/2 vs 3/2
xm = 1 + rng.standard_normal(2_000_000); nu_ = rng.choice([-1.0, 1.0], xm.size)
xv = (np.sqrt(10) * nu_ + rng.standard_normal(xm.size)) / np.sqrt(11)
val = 0.5 * np.mean(xm * xv**4)      # class -1 contributes 0 (E x_m = 0 there)
print(f"E[y x_m x_v^4] MC {val:.3f} vs (3+k4)/2 = {(3 - 200/121)/2:.3f}"); assert abs(val - (3 - 200/121)/2) < 0.02
ok("7.1.B.4: data-clone difference of E[y x_m x_v^4] is k4/2")

# ---------------------------------------------------------------- 7.2.A: whitening, kappa_4, cross covariance
r = np.sqrt(1 + bv)
gam = bv / (1 + bv + r)
assert abs((1 - gam) - 1 / r) < 1e-14
Nmc = 2_000_000
nu = rng.choice([-1.0, 1.0], Nmc); zeta = rng.standard_normal(Nmc)
s = (np.sqrt(bv) * nu + zeta) / r
k4 = np.mean(s**4) - 3 * np.mean(s**2) ** 2
print(f"var {s.var():.4f}, kappa_4 MC {k4:.4f}, formula {-2*bv**2/(1+bv)**2:.4f}")
assert abs(s.var() - 1) < 3e-3 and abs(k4 + 2 * bv**2 / (1 + bv) ** 2) < 2e-2
lam_ = rng.standard_normal(Nmc); nu_c = np.sign(lam_)
xu = np.sqrt(bu) * lam_ + rng.standard_normal(Nmc)
xv = (np.sqrt(bv) * nu_c + rng.standard_normal(Nmc)) / r
c_mc = np.mean(xu * xv); c_cf = np.sqrt(2 / np.pi) * np.sqrt(bu * bv / (1 + bv))
print(f"cross covariance MC {c_mc:.4f}, formula {c_cf:.4f}")
assert abs(c_mc - c_cf) < 1e-2 and abs(c_cf - 1.70) < 5e-3
ok("7.2.A: Q v = v/sqrt(1+beta_v), kappa_4 = -2 b^2/(1+b)^2, c = 1.70")


# ---------------------------------------------------------------- 7.2.D: single-neuron population loss (ReLU)
relu_g = lambda mu, s: mu * norm.cdf(mu / s) + s * norm.pdf(mu / s)   # E relu(N(mu, s^2))
sig2 = quad(lambda z: max(z, 0) * (z * z - 1) * norm.pdf(z), 0, np.inf)[0]
sig4 = quad(lambda z: max(z, 0) * (z**4 - 6 * z * z + 3) * norm.pdf(z), 0, np.inf)[0]
assert abs(sig2 - 1 / np.sqrt(2 * np.pi)) < 1e-10 and abs(sig4 + 1 / np.sqrt(2 * np.pi)) < 1e-10
k4v = -2 * bv**2 / (1 + bv) ** 2
c_x = np.sqrt(2 / np.pi) * np.sqrt(bu * bv / (1 + bv))
def L_relu(au, av, corr):
    sg = np.sqrt(1 - av**2 + av**2 / r**2); a_ = np.sqrt(bv) * av / r
    if not corr:
        s_ = np.sqrt(sg**2 + bu * au**2)
        Ep = 0.5 * (relu_g(a_, s_) + relu_g(-a_, s_))
    else:
        f = lambda l: relu_g(np.sqrt(bu) * au * l + a_ * np.sign(l), sg) * norm.pdf(l)
        Ep = quad(f, -np.inf, 0)[0] + quad(f, 0, np.inf)[0]
    return 1 - 0.5 * (Ep - relu_g(0.0, 1.0))
for av_ in (0.05, 0.1):
    ex, pr = L_relu(0, av_, 0) - L_relu(0, 0, 0), -sig4 * k4v / 48 * av_**4
    print(f"alpha_v={av_}: L-L0 {ex:.4e} vs -c04 alpha^4 {pr:.4e}"); assert abs(ex / pr - 1) < 0.02
for au_ in (0.02, 0.05):
    ex, pr = L_relu(au_, 0, 0) - L_relu(0, 0, 0), -sig2 * bu / 4 * au_**2
    print(f"alpha_u={au_}: L-L0 {ex:.4e} vs -c20 alpha^2 {pr:.4e}"); assert abs(ex / pr - 1) < 0.02
for e in (0.02, 0.04):
    ex, pr = (L_relu(e, e, 1) - L_relu(e, -e, 1)) / 2, -sig2 * c_x / 2 * e * e
    print(f"corr, alpha_u=alpha_v={e}: odd part {ex:.4e} vs -c11 alpha_u alpha_v {pr:.4e}"); assert abs(ex / pr - 1) < 0.02
ok("7.2.D: c20 = s2 b_u/4, c11 = s2 c/2, c04 = s4 k4/48 (ReLU, exact L)")

# ---------------------------------------------------------------- Bayes accuracies quoted in the text
Nb = 1_000_000
yb = rng.choice([-1, 1], Nb)
xa = np.where(yb > 0, bm, 0) + rng.standard_normal(Nb)
xb = np.where(yb > 0, np.sqrt(1 + bu), 1) * rng.standard_normal(Nb)
nb_ = rng.choice([-1.0, 1.0], Nb)
xc = np.where(yb > 0, mu_v * nb_ + s_v * rng.standard_normal(Nb), rng.standard_normal(Nb))
llr_m = norm.logpdf(xa, bm) - norm.logpdf(xa)
llr_u = norm.logpdf(xb, 0, np.sqrt(1 + bu)) - norm.logpdf(xb)
llr_v = np.log(0.5 * norm.pdf(xc, mu_v, s_v) + 0.5 * norm.pdf(xc, -mu_v, s_v)) - norm.logpdf(xc)
accs = [np.mean((L > 0) == (yb > 0)) for L in (llr_m, llr_m + llr_u, llr_m + llr_u + llr_v, llr_v)]
print("Bayes accuracies: mean only, mean+cov, all, v only:", np.round(accs, 4))
assert np.allclose(accs, [0.691, 0.767, 0.819, 0.714], atol=2.5e-3)
ok("Bayes accuracies 0.691 / 0.767 / 0.819 / 0.714")
print("all checks passed")
