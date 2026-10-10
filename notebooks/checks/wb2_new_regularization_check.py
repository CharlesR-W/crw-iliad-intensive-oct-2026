"""Numerical checks for notebooks/wb2_new_regularization.py (WB2 1.D opener, 1.D.3, 1.D.5).

Run: /home/crw/Programming/CellsDevelopmental/.venv/bin/python checks/wb2_new_regularization_check.py
Every check prints OK or raises AssertionError.
"""
import itertools

import numpy as np

rng = np.random.default_rng(0)


def ok(name, cond, info=""):
    assert cond, f"FAIL {name} {info}"
    print(f"OK  {name} {info}")


# ---------------------------------------------------------------- setup: ridge
n, p = 40, 6
X = rng.standard_normal((n, p))
y = rng.standard_normal(n)
lam = 0.3
ridge = np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
g = -2 / n * X.T @ (y - X @ ridge) + 2 * lam * ridge
ok("ridge closed form is stationary for (1/n)|y-Xb|^2+lam|b|^2", np.abs(g).max() < 1e-12)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
filt = Vt.T @ ((s**2 / (s**2 + n * lam)) * (U.T @ y) / s)
ok("ridge = SVD filter s^2/(s^2+n lam) on OLS coefficients", np.allclose(filt, ridge))


# ---------------------------------------------------------------- 1.D.3 lasso
def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0)


def lasso_ista(X, y, lam, iters=20000):
    n = X.shape[0]
    L = np.linalg.eigvalsh(X.T @ X / n).max()
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        b = soft(b - (X.T @ (X @ b - y) / n) / L, lam / L)
    return b


# orthonormal design X^T X = n I
Q, _ = np.linalg.qr(rng.standard_normal((n, p)))
Xo = np.sqrt(n) * Q
ok("orthonormal design has X^T X = nI", np.allclose(Xo.T @ Xo, n * np.eye(p)))
beta_true = np.array([2.0, -1.5, 0.4, 0.0, 0.1, -0.05])
yo = Xo @ beta_true + 0.5 * rng.standard_normal(n)
z = Xo.T @ yo / n
ok("z = X^T y / n equals OLS", np.allclose(z, np.linalg.lstsq(Xo, yo, rcond=None)[0]))
lam1 = 0.3
b_l = lasso_ista(Xo, yo, lam1)
ok("lasso (1/2n loss + lam|b|_1) = soft threshold of z", np.allclose(b_l, soft(z, lam1), atol=1e-10))
b_r = np.linalg.solve(Xo.T @ Xo + n * lam1 * np.eye(p), Xo.T @ yo)
ok("ridge with orthonormal design = z/(1+lam)", np.allclose(b_r, z / (1 + lam1)))


def l0_obj(b):
    return np.sum((yo - Xo @ b) ** 2) / (2 * n) + lam1**2 / 2 * np.count_nonzero(b)


best, bestval = None, np.inf
for mask in itertools.product([0, 1], repeat=p):
    S = np.array(mask, bool)
    b = np.zeros(p)
    if S.any():
        b[S] = np.linalg.lstsq(Xo[:, S], yo, rcond=None)[0]
    v = l0_obj(b)
    if v < bestval:
        best, bestval = b, v
ok("L0 penalty lam^2/2 |b|_0 gives hard threshold at lam", np.allclose(best, z * (np.abs(z) > lam1)))

# Laplace MAP: -log post = |y-Xb|^2/(2 sigma^2) + |b|_1/bscale  -> lasso with lam = sigma^2/(n bscale)
sigma, bscale = 0.5, 0.4
lam_map = sigma**2 / (n * bscale)
grid = np.linspace(-3, 3, 600001)
map_coords = []
for j in range(p):
    nlp = n / (2 * sigma**2) * (grid - z[j]) ** 2 + np.abs(grid) / bscale
    map_coords.append(grid[np.argmin(nlp)])
ok("Laplace MAP = soft threshold at sigma^2/(n b)", np.allclose(map_coords, soft(z, lam_map), atol=2e-5))

# general X: KKT/subgradient conditions and stability of zeros
Xg = rng.standard_normal((n, 10))
bg_true = np.r_[3, -2, 1.5, np.zeros(7)]
yg = Xg @ bg_true + rng.standard_normal(n)
lamg = 0.4
bg = lasso_ista(Xg, yg, lamg, iters=200000)
corr = Xg.T @ (yg - Xg @ bg) / n
act = np.abs(bg) > 1e-9
ok("KKT: active coords have x_j^T r/n = lam sign(b_j)", np.allclose(corr[act], lamg * np.sign(bg[act]), atol=1e-8))
ok("KKT: inactive coords have |x_j^T r/n| <= lam", np.all(np.abs(corr[~act]) <= lamg + 1e-8), f"(#zeros={np.sum(~act)})")
bg2 = lasso_ista(Xg, yg + 1e-3 * rng.standard_normal(n), lamg, iters=200000)
ok("zero pattern survives small perturbation of y", np.array_equal(np.abs(bg2) > 1e-9, act))

# posterior mean under Laplace prior is nonzero, with the sign of z
v = sigma**2 / n
for zz in [0.01, 0.05, 0.3, -0.2]:
    w = np.exp(-(grid - zz) ** 2 / (2 * v) - np.abs(grid) / bscale)
    pm = np.sum(grid * w) / np.sum(w)
    ok(f"Laplace posterior mean nonzero, sign(z) (z={zz})", np.sign(pm) == np.sign(zz) and abs(pm) > 1e-6,
       f"mean={pm:.4g}, MAP={soft(np.array(zz), lam_map):.3g}")

# ---------------------------------------------------------------- 1.D.5.1 input noise = ridge
s_in = 0.7
beta = rng.standard_normal(p)
# exact: E (y - (x+xi)^T b)^2 = (y - x^T b)^2 + s^2 |b|^2 ; check via Monte Carlo with many draws
M = 400000
xi = s_in * rng.standard_normal((M, p))
x0, y0 = X[0], y[0]
mc = np.mean((y0 - (x0 + xi) @ beta) ** 2)
exact = (y0 - x0 @ beta) ** 2 + s_in**2 * beta @ beta
ok("Gaussian input noise adds s^2|b|^2 (Monte Carlo)", abs(mc - exact) / exact < 1e-2, f"mc={mc:.4f} exact={exact:.4f}")
# minimizer of expected objective = ridge with lam = s^2
b_noise = np.linalg.solve(X.T @ X / n + s_in**2 * np.eye(p), X.T @ y / n)
b_ridge = np.linalg.solve(X.T @ X + n * s_in**2 * np.eye(p), X.T @ y)
ok("noisy-input minimizer = ridge with lam = s^2", np.allclose(b_noise, b_ridge))
# anisotropic noise covariance Sigma_xi -> penalty b^T Sigma_xi b
A = rng.standard_normal((p, p))
Sxi = A @ A.T / p
xi = rng.multivariate_normal(np.zeros(p), Sxi, size=M)
mc = np.mean((y0 - (x0 + xi) @ beta) ** 2)
exact = (y0 - x0 @ beta) ** 2 + beta @ Sxi @ beta
ok("anisotropic input noise adds b^T Sigma_xi b", abs(mc - exact) / exact < 1e-2)

# ---------------------------------------------------------------- 1.D.5.2 nonlinear second-order expansion
# f(x) = tanh(w.x + c) + 0.3 sin(u.x) in d=2; expectation by Gauss-Hermite quadrature
d = 2
w_, c_, u_ = np.array([1.3, -0.7]), 0.2, np.array([0.5, 1.1])


def f(x):
    return np.tanh(x @ w_ + c_) + 0.3 * np.sin(x @ u_)


def grad_f(x):
    t = np.tanh(x @ w_ + c_)
    return (1 - t**2) * w_ + 0.3 * np.cos(x @ u_) * u_


def lap_f(x):
    t = np.tanh(x @ w_ + c_)
    return -2 * t * (1 - t**2) * (w_ @ w_) - 0.3 * np.sin(x @ u_) * (u_ @ u_)


gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60)  # weight exp(-x^2/2)
gh_w = gh_w / gh_w.sum()
XX, YY = np.meshgrid(gh_x, gh_x, indexing="ij")
WW = np.outer(gh_w, gh_w)
pts = np.stack([XX.ravel(), YY.ravel()], 1)
x_pt, y_pt = np.array([0.3, -0.4]), 0.9
r = y_pt - f(x_pt)
errs = []
for s_ in [0.2, 0.1, 0.05, 0.025]:
    E = np.sum(WW.ravel() * (y_pt - f(x_pt + s_ * pts)) ** 2)
    approx = r**2 + s_**2 * (grad_f(x_pt) @ grad_f(x_pt) - r * lap_f(x_pt))
    errs.append(abs(E - approx))
ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
ok("E(y-f(x+xi))^2 = r^2 + s^2(|grad f|^2 - r Lap f) + O(s^4)", (all(q > 10 for q in ratios) and 15 < ratios[-1] < 17),
   f"error ratios on halving s: {np.round(ratios, 2)}")

# ---------------------------------------------------------------- 1.D.5.3 the circle: noise vs Tikhonov
N = 4096
t = 2 * np.pi * np.arange(N) / N
m = np.sin(t) + 0.5 * np.cos(3 * t) + 0.2 * np.sin(5 * t + 1)
k = np.fft.fftfreq(N, d=1 / N)
mh = np.fft.fft(m)
m2 = np.real(np.fft.ifft(-(k**2) * mh))  # m''
for s_ in [0.1, 0.05]:
    # exact minimizer of noisy objective: convolution of m with wrapped Gaussian (direct, not via Fourier formula)
    dx = 2 * np.pi / N
    off = (np.arange(N) - N // 2) * dx
    ker = sum(np.exp(-((off + 2 * np.pi * j) ** 2) / (2 * s_**2)) for j in (-1, 0, 1))
    ker = ker / ker.sum()
    f_noise = np.real(np.fft.ifft(np.fft.fft(np.fft.ifftshift(ker)) * mh))
    ok(f"circle: noisy-training optimum has multiplier exp(-s^2k^2/2) (s={s_})",
       np.allclose(f_noise, np.real(np.fft.ifft(np.exp(-s_**2 * k**2 / 2) * mh)), atol=1e-10))
    f_tik = np.real(np.fft.ifft(mh / (1 + s_**2 * k**2)))
    # check Tikhonov multiplier is the minimizer: gradient of int (m-f)^2 + s^2 int f'^2 vanishes
    fp = np.real(np.fft.ifft(-(k**2) * np.fft.fft(f_tik)))
    ok(f"circle: Tikhonov optimum satisfies f - s^2 f'' = m (s={s_})", np.allclose(f_tik - s_**2 * fp, m, atol=1e-10))
    e_noise = np.abs(f_noise - (m + s_**2 / 2 * m2)).max()
    e_tik = np.abs(f_tik - (m + s_**2 * m2)).max()
    gap = np.abs(f_noise - f_tik).max() / (s_**2 * np.abs(m2).max())
    ok(f"circle: noise ~ m + s^2/2 m'', Tikhonov ~ m + s^2 m'' (s={s_})", e_noise < 3 * s_**4 * 700 and e_tik < 3 * s_**4 * 700,
       f"residuals {e_noise:.2e}, {e_tik:.2e}; gap/(s^2 max|m''|)={gap:.3f}")
    # full second-order objective int (m-f)^2 - s^2 int m f''  -> minimizer m + s^2/2 m''
    f_full = m + s_**2 / 2 * m2
    gfull = -2 * (m - f_full) - s_**2 * m2
    ok(f"circle: full 2nd-order objective is stationary at m + s^2/2 m'' (s={s_})", np.abs(gfull).max() < 1e-10)

# non-uniform p on R: E[m(x) | x+xi] = m + s^2 (m''/2 + (log p)' m') + O(s^4)
xs = np.linspace(-12, 12, 24001)
dxs = xs[1] - xs[0]
logp = -xs**2 / 2 - 0.3 * np.sin(2 * xs)
pdens = np.exp(logp)
pdens /= pdens.sum() * dxs
mfun = lambda x: np.sin(1.3 * x) + 0.2 * x**2  # noqa: E731
dm = lambda x: 1.3 * np.cos(1.3 * x) + 0.4 * x  # noqa: E731
d2m = lambda x: -1.69 * np.sin(1.3 * x) + 0.4  # noqa: E731
dlogp = lambda x: -x - 0.6 * np.cos(2 * x)  # noqa: E731
errs = []
for s_ in [0.1, 0.05, 0.025]:
    tt = 0.4
    wgt = pdens * np.exp(-((tt - xs) ** 2) / (2 * s_**2))
    post = np.sum(wgt * mfun(xs)) / np.sum(wgt)
    pred = mfun(tt) + s_**2 * (d2m(tt) / 2 + dlogp(tt) * dm(tt))
    errs.append(abs(post - pred))
ok("E[m(x)|x+xi] = m + s^2(m''/2 + (log p)' m') + O(s^4)", errs[0] / errs[1] > 12 and errs[1] / errs[2] > 12,
   f"errors {np.array(errs)}")

# ---------------------------------------------------------------- 1.D.5.4 mirror augmentation
pp = 6
S = np.eye(pp)[::-1]  # reverse coordinates: orthogonal involution
P, Qm = (np.eye(pp) + S) / 2, (np.eye(pp) - S) / 2
ok("P, Q are complementary orthogonal projections", np.allclose(P @ P, P) and np.allclose(Qm @ Qm, Qm)
   and np.allclose(P @ Qm, 0) and np.allclose(P + Qm, np.eye(pp)))
Xa = rng.standard_normal((n, pp))
ya = rng.standard_normal(n)
b = rng.standard_normal(pp)
# augmented inputs are S x_i, i.e. rows of X S^T; prediction b^T S x_i
aug = (np.sum((ya - Xa @ b) ** 2) + np.sum((ya - (Xa @ S.T) @ b) ** 2)) / (2 * n)
dec = np.sum((ya - Xa @ P @ b) ** 2) / n + np.sum((Xa @ Qm @ b) ** 2) / n
ok("mirror augmentation = (1/n)|y-XPb|^2 + (1/n)|XQb|^2", np.isclose(aug, dec))
Xaug = np.vstack([Xa, Xa @ S.T])
yaug = np.r_[ya, ya]
baug = np.linalg.lstsq(Xaug, yaug, rcond=None)[0]
ok("augmented ERM has Q b = 0 (invariant)", np.allclose(Qm @ baug, 0, atol=1e-10))
bsym = np.linalg.lstsq(Xa @ P, ya, rcond=None)[0]  # min-norm -> lies in range(P)
ok("augmented ERM = OLS on symmetrized inputs P x_i", np.allclose(baug, bsym))

# general finite group: cyclic shifts of R^pp
shifts = [np.roll(np.eye(pp), kk, axis=0) for kk in range(pp)]
Pg = sum(shifts) / pp
ok("group average P is a symmetric idempotent", np.allclose(Pg, Pg.T) and np.allclose(Pg @ Pg, Pg))
x1, y1 = Xa[0], ya[0]
lhs = np.mean([(y1 - b @ gm @ x1) ** 2 for gm in shifts])
pen = np.mean([(b @ (gm - Pg) @ x1) ** 2 for gm in shifts])
ok("orbit-averaged loss = (y - b^T P x)^2 + mean_g (b^T (g-P) x)^2", np.isclose(lhs, (y1 - b @ Pg @ x1) ** 2 + pen))
bperp = (np.eye(pp) - Pg) @ b
pen2 = np.mean([(bperp @ (gm - Pg) @ x1) ** 2 for gm in shifts])
ok("penalty depends only on (I-P)b", np.isclose(pen, pen2))
binv = Pg @ b
ok("penalty vanishes for invariant b", np.isclose(np.mean([(binv @ (gm - Pg) @ x1) ** 2 for gm in shifts]), 0))

# ---------------------------------------------------------------- 1.D.5.5 partial mirror augmentation
# S-invariant input distribution: empirical law of {x_i} U {S x_i} (labels arbitrary, independent draws)
Xh = rng.standard_normal((n, pp))
Xpop = np.vstack([Xh, Xh @ S.T])
ypop = rng.standard_normal(2 * n) + Xpop @ rng.standard_normal(pp)
Sig = Xpop.T @ Xpop / (2 * n)
ok("S-invariant inputs: P Sigma Q = 0", np.allclose(P @ Sig @ Qm, 0))
b_clean = np.linalg.solve(Sig, Xpop.T @ ypop / (2 * n))
for alpha in [0.1, 0.25, 0.4]:
    # augmented risk: weight (1-alpha) on (x,y), alpha on (Sx,y); minimize by weighted least squares
    Xw = np.vstack([Xpop, Xpop @ S.T])
    yw = np.r_[ypop, ypop]
    ww = np.r_[np.full(2 * n, 1 - alpha), np.full(2 * n, alpha)]
    b_aug = np.linalg.solve(Xw.T @ (ww[:, None] * Xw), Xw.T @ (ww * yw))
    rho = 1 - 2 * alpha
    ok(f"partial aug (alpha={alpha}): P b unchanged, Q b shrunk by 1-2alpha",
       np.allclose(P @ b_aug, P @ b_clean) and np.allclose(Qm @ b_aug, rho * Qm @ b_clean))
    kappa = 2 * alpha / (1 - 2 * alpha)
    b_gr = np.linalg.solve(Sig + kappa * Qm @ Sig @ Qm, Xpop.T @ ypop / (2 * n))
    ok(f"partial aug (alpha={alpha}) = clean risk + kappa b^T Q Sigma Q b, kappa=2a/(1-2a)", np.allclose(b_gr, b_aug))

print("\nall checks passed")
