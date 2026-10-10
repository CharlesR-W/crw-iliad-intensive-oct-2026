"""Checks for double_descent.html.

Model (Hastie, Montanari, Rosset, Tibshirani 2019, arXiv 1903.08560):
  x ~ N(0, I_p), y = x^T beta + eps, eps ~ N(0, sigma^2), ||beta||^2 = r^2, p/n -> gamma.
  Risk  R = E[(x0^T bhat - x0^T beta)^2 | X] = E||bhat - beta||^2   (EXCLUDES the irreducible sigma^2;
  this is their definition of R_X, eq. (2)).
  Ridge: bhat_lam = argmin (1/n)||y - X b||^2 + lam ||b||^2 = (X^T X + n lam I)^{-1} X^T y.

Ridge asymptotic risk (their Thm 5 / Dobriban-Wager 2018), with m(z) the Stieltjes transform of the
Marchenko-Pastur law of  S = X^T X / n  (aspect ratio gamma, includes the atom at 0 when gamma > 1):
  bias^2 = r^2 lam^2 m'(-lam)            [ = (r^2/p) tr lam^2 (S + lam)^-2 ]
  var    = sigma^2 gamma (m(-lam) - lam m'(-lam))   [ = (sigma^2/n) tr S (S + lam)^-2 ]
  m solves  gamma z m^2 - (1 - gamma - z) m + 1 = 0.
Optimal lam* = sigma^2 gamma / r^2, where R* = sigma^2 gamma m(-lam*).

Run:  python double_descent_check.py   (writes ../double_descent_mc.json is NOT done; data is printed
for embedding into the html via --emit).
"""
import json
import sys

import numpy as np
from scipy import integrate, optimize


# ---------- formulas (mirrored in the JS) ----------
def m_mp(lam, g):
    """Stieltjes transform m(z) at z = -lam < 0 for MP(gamma), variance 1."""
    a = 1 - g + lam
    return (-a + np.sqrt(a * a + 4 * g * lam)) / (2 * g * lam)


def mprime_mp(lam, g):
    """m'(z) at z = -lam, from implicit differentiation of gamma z m^2 - (1-gamma-z) m + 1 = 0."""
    m = m_mp(lam, g)
    z = -lam
    return (g * m * m + m) / ((1 - g - z) - 2 * g * z * m)


def ridge_bv(lam, g, r2, s2):
    m = m_mp(lam, g)
    mp = mprime_mp(lam, g)
    return r2 * lam * lam * mp, s2 * g * (m - lam * mp)


def minnorm_bv(g, r2, s2):
    if g < 1:
        return 0.0, s2 * g / (1 - g)
    return r2 * (1 - 1 / g), s2 / (g - 1)


# ---------- independent checks ----------
def mp_density(s, g):
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    if s <= lo or s >= hi:
        return 0.0
    return np.sqrt((hi - s) * (s - lo)) / (2 * np.pi * g * s)


def m_by_quad(lam, g, power=1):
    lo, hi = (1 - np.sqrt(g)) ** 2, (1 + np.sqrt(g)) ** 2
    val, _ = integrate.quad(lambda s: mp_density(s, g) / (s + lam) ** power, lo, hi, limit=200)
    if g > 1:  # atom at 0 with mass 1 - 1/g
        val += (1 - 1 / g) / lam ** power
    return val


def mc_risk(n, g, lam, r2, s2, reps, rng):
    """Direct simulation: draw X, beta (uniform on sphere radius r), eps; fit; risk = ||bhat-beta||^2."""
    p = int(round(g * n))
    out = []
    for _ in range(reps):
        X = rng.standard_normal((n, p))
        b = rng.standard_normal(p)
        b *= np.sqrt(r2) / np.linalg.norm(b)
        y = X @ b + np.sqrt(s2) * rng.standard_normal(n)
        if lam == 0:
            bh = np.linalg.pinv(X) @ y
        else:
            bh = np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
        out.append(np.sum((bh - b) ** 2))
    out = np.array(out)
    return out.mean(), out.std() / np.sqrt(reps)


def main():
    rng = np.random.default_rng(0)
    ok = True

    print("1) Stieltjes closed form vs quadrature of MP density (+atom)")
    for g in [0.3, 0.9, 1.0, 1.7, 3.0]:
        for lam in [0.01, 0.3, 2.0]:
            m_cf, m_q = m_mp(lam, g), m_by_quad(lam, g)
            mp_cf, mp_q = mprime_mp(lam, g), m_by_quad(lam, g, power=2)
            e = max(abs(m_cf - m_q) / m_q, abs(mp_cf - mp_q) / mp_q)
            ok &= e < 1e-5
            print(f"  g={g:4} lam={lam:5}: m={m_cf:.6f} (quad {m_q:.6f})  m'={mp_cf:.6f} (quad {mp_q:.6f})  rel err {e:.1e}")

    print("2) ridge lam->0 limit equals min-norm formula (Hastie Thm 1)")
    for g in [0.5, 0.8, 1.5, 3.0]:
        b0, v0 = ridge_bv(1e-7, g, 2.0, 1.0)
        b1, v1 = minnorm_bv(g, 2.0, 1.0)
        print(f"  g={g}: ridge(1e-7) bias={b0:.5f} var={v0:.5f}   minnorm bias={b1:.5f} var={v1:.5f}")
        ok &= abs(b0 - b1) < 1e-4 and abs(v0 - v1) < 1e-4

    print("3) optimal lam by numerical minimization vs sigma^2 gamma / r^2")
    for g, r2, s2 in [(0.5, 1.0, 1.0), (2.0, 4.0, 1.0), (1.0, 2.0, 0.5)]:
        res = optimize.minimize_scalar(lambda L: sum(ridge_bv(np.exp(L), g, r2, s2)), bounds=(-12, 6), method="bounded")
        lam_num, lam_th = np.exp(res.x), s2 * g / r2
        Rstar = s2 * g * m_mp(lam_th, g)
        print(f"  g={g} r2={r2} s2={s2}: argmin={lam_num:.5f}  theory={lam_th:.5f}  R*={sum(ridge_bv(lam_th,g,r2,s2)):.5f} = s2*g*m = {Rstar:.5f}")
        ok &= abs(lam_num / lam_th - 1) < 1e-3 and abs(Rstar - sum(ridge_bv(lam_th, g, r2, s2))) < 1e-10

    print("4) direct Monte Carlo (n=400, 60 reps) vs asymptotic formula")
    for g, lam, r2, s2 in [(0.5, 0.0, 1.0, 1.0), (0.8, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0),
                           (2.0, 0.5, 1.0, 1.0), (0.5, 0.1, 4.0, 1.0), (1.0, 0.25, 4.0, 1.0)]:
        th = sum(minnorm_bv(g, r2, s2)) if lam == 0 else sum(ridge_bv(lam, g, r2, s2))
        mean, se = mc_risk(400, g, lam, r2, s2, 60, rng)
        z = (mean - th) / se
        print(f"  g={g} lam={lam} r2={r2}: MC {mean:.4f} +- {se:.4f}   theory {th:.4f}   rel diff {(mean-th)/th:+.3f}  z={z:+.2f}")
    # finite-n exact for g<1, lam=0: sigma^2 p/(n-p-1)
    print("   (finite-n exact, g<1 min-norm: s2 p/(n-p-1) = %.4f at g=0.5, %.4f at g=0.8)" % (200 / 199, 320 / 79))

    print("5) reference values for the JS self-test")
    ref = []
    for g, lam, r2, s2 in [(0.5, 0.1, 1.0, 1.0), (2.0, 0.5, 4.0, 1.0), (1.0, 0.01, 1.0, 1.0), (3.0, 1.5, 2.0, 1.0)]:
        b, v = ridge_bv(lam, g, r2, s2)
        ref.append([g, lam, r2, s2, b, v])
        print(f"  ridge g={g} lam={lam} r2={r2} s2={s2}: bias={b:.10f} var={v:.10f}")
    print("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")

    if "--emit" in sys.argv:
        # finite-n data: n=100, nonzero eigenvalues of X^T X / n for each p and seed.
        n, seeds = 100, 8
        ps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 105, 110, 120, 140, 160, 180, 200, 230, 260, 300, 350, 400]
        data = {"n": n, "seeds": seeds, "ps": ps, "eig": []}
        r2 = np.random.default_rng(12345)
        for p in ps:
            row = []
            for _ in range(seeds):
                X = r2.standard_normal((n, p))
                s = np.linalg.svd(X, compute_uv=False) ** 2 / n  # min(n,p) nonzero eigs
                row.append([float(f"{v:.5g}") for v in s])
            data["eig"].append(row)
        with open(sys.argv[sys.argv.index("--emit") + 1], "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print("wrote", sys.argv[sys.argv.index("--emit") + 1])
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
