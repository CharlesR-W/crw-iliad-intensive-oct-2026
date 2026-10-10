"""Checks for spectral_filters.html (WB2 §3.3 + WB5 §1).

Panel A: filter factors.
  loss  L(b) = (1/2n)||X b - y||^2  (+ (lam/2)||b||^2 for ridge)
  ridge  : b = (X^T X + n lam I)^{-1} X^T y  ->  phi_k = s_k^2/(s_k^2 + n lam)
  GF     : db/dt = -(1/n) X^T (X b - y), b(0)=0 -> phi_k = 1 - exp(-s_k^2 t / n)
  checked against direct linear algebra and small-step gradient descent.

Panel B: ridge on Gaussian features x ~ N(0, diag(lam_k)), lam_k = k^-alpha,
  y = x.beta + eps, eps ~ N(0, sigma2), target power c_k^2 = lam_k beta_k^2 ∝ k^-alpha'.
  Estimator minimises (1/n)||X b - y||^2 + lam ||b||^2  (same n*lam convention as A).
  Deterministic equivalent (Hastie et al. 2022; Bach 2023; Simon et al. 2023):
    kappa solves  kappa (1 - df1(kappa)/n) = lam,  df1 = sum lam_k/(lam_k+kappa)
    df2 = sum lam_k^2/(lam_k+kappa)^2,  B = kappa^2 sum c_k^2/(lam_k+kappa)^2
    excess risk  E = (B + sigma2)/(1 - df2/n) - sigma2
  Heuristic: E_heur(n) = min_{K<=n} [ sum_{k>K} c_k^2 + sigma2 K / n ].
  Checked against Monte Carlo ridge for two (alpha, alpha', sigma2, lam) settings,
  and the infinite-spectrum quadrature used in the JS is checked against a brute sum.
Run:  python spectral_filters_check.py   (writes spectral_filters_mc.json next to it)
"""
import json, math, os
import numpy as np
from scipy.special import zeta

rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- Panel A
def check_filters():
    n, p = 400, 30
    X = rng.standard_normal((n, p)) * np.linspace(2.0, 0.02, p)  # spread of s_k^2
    y = rng.standard_normal(n)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2 = s**2
    bols = Vt.T @ ((U.T @ y) / s)
    out = {}
    lam = 1e-3
    br = np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
    phi_r_emp = (Vt @ br) / (Vt @ bols)
    out["ridge_maxerr"] = float(np.max(np.abs(phi_r_emp - s2 / (s2 + n * lam))))
    # gradient descent with tiny step approximates gradient flow
    t, steps = 50.0, 200000
    eta = t / steps
    b = np.zeros(p)
    for _ in range(steps):
        b -= eta * (X.T @ (X @ b - y)) / n
    phi_g_emp = (Vt @ b) / (Vt @ bols)
    phi_g_th = 1 - np.exp(-s2 * t / n)
    # discrete GD exact filter: 1-(1-eta s2/n)^steps
    phi_g_disc = 1 - (1 - eta * s2 / n) ** steps
    out["gf_maxerr_vs_flow"] = float(np.max(np.abs(phi_g_emp - phi_g_th)))
    out["gf_maxerr_vs_discrete"] = float(np.max(np.abs(phi_g_emp - phi_g_disc)))
    # half points
    out["ridge_half"] = float((n * lam) / (n * lam + n * lam))
    tt = math.log(2) / lam
    out["gf_half_at_nlam_with_t=ln2/lam"] = float(1 - math.exp(-(n * lam) * tt / n))
    return out

# ---------------------------------------------------------------- Panel B
def spectrum(alpha, ap, p):
    k = np.arange(1, p + 1, dtype=float)
    lk = k**-alpha
    ck2 = k**-ap / zeta(ap)          # normalised so sum_{k=1}^inf c_k^2 = 1
    return lk, ck2

def de_sums(lk, ck2, kappa, w=None):
    w = np.ones_like(lk) if w is None else w
    r = lk / (lk + kappa)
    df1 = np.sum(w * r)
    df2 = np.sum(w * r * r)
    B = kappa**2 * np.sum(w * ck2 / (lk + kappa) ** 2)
    return df1, df2, B

def solve_kappa(lam, n, sums):
    # f(kappa) = kappa - lam - kappa*df1/n, increasing; root in (lam, ...)
    lo, hi = max(lam, 1e-300), 1.0
    def f(k):
        return k * (1 - sums(k)[0] / n) - lam
    while f(hi) < 0:
        hi *= 10
    # lower bracket: kappa0 where df1 = n (lam=0) or lam
    lo = 1e-30
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-12:
            break
    return math.sqrt(lo * hi)

def de_risk(lam, n, sigma2, sums):
    kap = solve_kappa(lam, n, sums)
    df1, df2, B = sums(kap)
    return (B + sigma2) / (1 - df2 / n) - sigma2

# --- infinite spectrum quadrature, same as the JS
K0 = 1000
def make_inf_sums(alpha, ap, nq=500, umax=math.log(1e10)):
    kk = np.arange(1, K0 + 1, dtype=float)
    lk_e = kk**-alpha
    ck_e = kk**-ap / zeta(ap)
    u = np.linspace(math.log(K0 + 0.5), umax, nq)
    x = np.exp(u)
    wq = np.full(nq, u[1] - u[0]) * x
    wq[0] *= 0.5; wq[-1] *= 0.5
    lk_q = x**-alpha
    ck_q = x**-ap / zeta(ap)
    Xmax = math.exp(umax)
    lk = np.concatenate([lk_e, lk_q]); ck = np.concatenate([ck_e, ck_q])
    w = np.concatenate([np.ones(K0), wq])
    def sums(kappa):
        df1, df2, B = de_sums(lk, ck, kappa, w)
        # analytic tail x > Xmax where lam_x << kappa
        df1 += Xmax ** (1 - alpha) / (alpha - 1) / kappa
        df2 += Xmax ** (1 - 2 * alpha) / (2 * alpha - 1) / kappa**2
        B += Xmax ** (1 - ap) / (ap - 1) / zeta(ap)
        return df1, df2, B
    return sums

def heuristic(n, sigma2, ap):
    # min over K<=n of tail(K) + sigma2 K/n ; tail(K)= sum_{k>K} c_k^2
    Ks = np.unique(np.round(np.logspace(0, math.log10(n), 400)).astype(int))
    tail = np.array([zeta(ap, K + 1) / zeta(ap) for K in Ks])
    return float(np.min(tail + sigma2 * Ks / n))

def mc_ridge(alpha, ap, sigma2, lam, n, p, reps, ntest=4000):
    lk, ck2 = spectrum(alpha, ap, p)
    sd = np.sqrt(lk)
    beta = np.sqrt(ck2 / lk)  # c_k^2 = lam_k beta_k^2, all signs +
    errs = []
    for _ in range(reps):
        X = rng.standard_normal((n, p)) * sd
        y = X @ beta + math.sqrt(sigma2) * rng.standard_normal(n)
        if lam > 0:
            if n < p:  # dual form
                a = np.linalg.solve(X @ X.T + n * lam * np.eye(n), y)
                b = X.T @ a
            else:
                b = np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
        else:
            b = np.linalg.pinv(X) @ y
        d = b - beta
        errs.append(float(np.sum(lk * d * d)))  # exact excess risk E_x[(x.d)^2]
    return float(np.mean(errs)), float(np.std(errs) / math.sqrt(reps))

def opt_lambda_de(n, sigma2, sums, grid=np.logspace(-9, 0, 91)):
    best = (None, np.inf)
    for lam in grid:
        r = de_risk(lam, n, sigma2, sums)
        if r < best[1]:
            best = (lam, r)
    return best

def main():
    print("=== Panel A: filter factors ===")
    for k, v in check_filters().items():
        print(f"  {k:36s} {v:.3e}")

    print("\n=== Panel B: quadrature (infinite spectrum) vs brute sum to 2e7 + tail ===")
    for (alpha, ap) in [(1.5, 2.0), (1.1, 1.3), (2.5, 4.0)]:
        sq = make_inf_sums(alpha, ap)
        P = 2_000_000
        lk, ck = spectrum(alpha, ap, P)
        def brute(kap):
            df1, df2, B = de_sums(lk, ck, kap)
            df1 += P ** (1 - alpha) / (alpha - 1) / kap  # tail beyond P (lam<<kap)
            df2 += P ** (1 - 2 * alpha) / (2 * alpha - 1) / kap**2
            B += zeta(ap, P + 1) / zeta(ap)
            return df1, df2, B
        for kap in [1e-2, 1e-4, 1e-6]:
            a, b = np.array(sq(kap)), np.array(brute(kap))
            print(f"  alpha={alpha} a'={ap} kappa={kap:.0e}  rel err (df1,df2,B) = "
                  + ", ".join(f"{x:.1e}" for x in np.abs(a - b) / np.abs(b)))

    print("\n=== Panel B: deterministic equivalent vs Monte Carlo ridge (p=3000) ===")
    p = 3000
    settings = [
        dict(alpha=1.5, ap=2.0, sigma2=0.0, lam=1e-7, tag="noiseless, tiny ridge"),
        dict(alpha=1.5, ap=2.0, sigma2=1e-2, lam=None, tag="sigma2=1e-2, optimal ridge"),
        dict(alpha=2.0, ap=1.5, sigma2=1e-3, lam=None, tag="alpha=2, a'=1.5, opt ridge"),
    ]
    mc_json = {}
    for s in settings:
        lk, ck2 = spectrum(s["alpha"], s["ap"], p)
        sums_p = lambda kap, lk=lk, ck2=ck2: de_sums(lk, ck2, kap)
        sums_inf = make_inf_sums(s["alpha"], s["ap"])
        print(f"  [{s['tag']}] alpha={s['alpha']} a'={s['ap']} sigma2={s['sigma2']}")
        rows = []
        for n in [25, 50, 100, 200, 400, 800]:
            lam = s["lam"]
            if lam is None:
                lam, _ = opt_lambda_de(n, s["sigma2"], sums_p)
            de_p = de_risk(lam, n, s["sigma2"], sums_p)
            de_inf = de_risk(lam, n, s["sigma2"], sums_inf)
            mc, se = mc_ridge(s["alpha"], s["ap"], s["sigma2"], lam, n, p, reps=40)
            print(f"    n={n:4d} lam={lam:.1e}  MC={mc:.4e}±{se:.1e}  DE(p)={de_p:.4e}"
                  f"  DE(inf)={de_inf:.4e}  MC/DE(p)={mc/de_p:.3f}")
            rows.append(dict(n=n, lam=lam, mc=mc, se=se, de=de_p))
        key = f"{s['alpha']}_{s['ap']}_{s['sigma2']}"
        mc_json[key] = dict(alpha=s["alpha"], ap=s["ap"], sigma2=s["sigma2"], p=p,
                            ridge="tiny" if s["lam"] else "optimal", rows=rows)

    print("\n=== Panel B: asymptotic slopes (DE, infinite spectrum) vs prediction ===")
    for (alpha, ap, sigma2) in [(1.5, 2.0, 0.0), (1.5, 3.0, 0.0), (1.2, 1.5, 0.0),
                                (1.5, 2.0, 1e-2), (2.0, 3.0, 1e-2), (1.5, 5.0, 0.0)]:
        sums = make_inf_sums(alpha, ap)
        ns = [1e4, 1e5]
        if sigma2 == 0:
            r = [de_risk(1e-300 if False else 0.0, n, 0.0, sums) for n in ns]
        else:
            r = [opt_lambda_de(n, sigma2, sums, np.logspace(-12, 0, 241))[1] for n in ns]
        h = [heuristic(int(n), sigma2, ap) for n in ns]
        slope = -math.log(r[1] / r[0]) / math.log(10)
        hs = -math.log(h[1] / h[0]) / math.log(10)
        s0 = min(ap - 1, 2 * alpha)
        pred = s0 if sigma2 == 0 else s0 / (s0 + 1)
        print(f"  alpha={alpha} a'={ap} sigma2={sigma2:g}: DE slope={slope:.3f} "
              f"heuristic slope={hs:.3f} predicted={pred:.3f}")

    # values for JS cross-check
    js = {}
    for (alpha, ap, sigma2) in [(1.5, 2.0, 0.0), (1.5, 2.0, 1e-2)]:
        sums = make_inf_sums(alpha, ap)
        for n in [10, 100, 1000, 10000]:
            if sigma2 == 0:
                v = de_risk(0.0, n, 0.0, sums)
            else:
                v = opt_lambda_de(n, sigma2, sums, np.logspace(-12, 0, 481))[1]
            js[f"{alpha}_{ap}_{sigma2}_{n}"] = dict(de=v, heur=heuristic(n, sigma2, ap))
    print("\nJS reference values:", json.dumps(js, indent=None))
    with open(os.path.join(HERE, "spectral_filters_mc.json"), "w") as f:
        json.dump(dict(mc=mc_json, js_ref=js), f, indent=1)

if __name__ == "__main__" and not os.environ.get("EMBED"):
    main()


# ---------------------------------------------------------------- embedded MC
def embed_mc(p=50_000, reps=16):
    """MC points shown in the figure at its default settings. p=5e4 >> n so the
    finite-p truncation is negligible; lambda = DE-optimal for the infinite spectrum
    (the same curve the JS draws)."""
    out = []
    for (alpha, ap, sigma2) in [(1.5, 2.0, 0.0), (1.5, 2.0, 1e-2)]:
        sums = make_inf_sums(alpha, ap)
        rows = []
        for n in [16, 32, 64, 128, 256, 512]:
            lam, de = opt_lambda_de(n, sigma2, sums, np.logspace(-10, 0, 201))
            mc, se = mc_ridge(alpha, ap, sigma2, lam, n, p, reps)
            print(f"  embed a={alpha} a'={ap} s2={sigma2} n={n} lam={lam:.1e} "
                  f"MC={mc:.4e}±{se:.1e} DE(inf)={de:.4e} ratio={mc/de:.3f}")
            rows.append(dict(n=n, mc=mc, se=se, de=de, lam=lam))
        out.append(dict(alpha=alpha, ap=ap, sigma2=sigma2, p=p, rows=rows))
    return out

if __name__ == "__main__" and os.environ.get("EMBED"):
    res = embed_mc()
    with open(os.path.join(HERE, "spectral_filters_embed.json"), "w") as f:
        json.dump(res, f)
