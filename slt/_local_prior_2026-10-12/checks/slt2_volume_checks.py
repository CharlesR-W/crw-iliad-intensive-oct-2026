#!/usr/bin/env python3
"""Symbolic and numerical checks for every formula stated in slt2_volume (SLT NB2).

Run: python slt/checks/slt2_volume_checks.py   (CPU, well under a minute). One line per
check; exits non-zero on any failure. Notation as in the notebook: prior phi, K >= 0,
V(t) = prior mass of {K < t}, v = V', Zbar_n = int phi e^{-nK}, zeta(z) = int K^z phi.
"""
from __future__ import annotations

import math
import sys

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate, special, optimize

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


mp.mp.dps = 30
t, z, n, s, a, b, w = sp.symbols("t z n s a b w", positive=True)

# ------------------------------------------------------------------ §1 volumes (uniform prior on [-1,1]^d)
# K = w^(2k), d = 1: V = t^(1/(2k))
for k in (1, 2, 3):
    V = mp.quad(lambda x: 0.5, [-mp.mpf(0.01) ** (1 / (2 * k)), mp.mpf(0.01) ** (1 / (2 * k))])
    check(f"V(t) for w^{2*k} is t^(1/{2*k})", abs(V - mp.mpf(0.01) ** (1 / (2 * k))) < 1e-20)

# K = x^2 + y^4: V = (t^{3/4}/2) * int_{-1}^1 sqrt(1-s^4) ds ; prior density 1/4
I4 = mp.quad(lambda u: mp.sqrt(1 - u ** 4), [-1, 1])
for tt in (1e-2, 1e-4):
    tt = mp.mpf(tt)
    V = mp.quad(lambda y: 0.25 * 2 * mp.sqrt(tt - y ** 4), [-tt ** 0.25, tt ** 0.25])
    check(f"V(t) for x^2+y^4 = c t^(3/4), t={float(tt)}", abs(V / (tt ** 0.75 * I4 / 2) - 1) < 1e-12,
          f"c = {float(I4/2):.4f}")

# K = (ab)^2: V(t) = sqrt(t) (1 - log(t)/2) for t<1; v(t) = (1/4) t^{-1/2} (-log t)
Vab = sp.sqrt(t) * (1 - sp.log(t) / 2)
check("v = V' for (ab)^2 is (1/4) t^(-1/2)(-log t)",
      sp.simplify(sp.diff(Vab, t) - sp.Rational(1, 4) * t ** sp.Rational(-1, 2) * (-sp.log(t))) == 0)
for tt in (1e-2, 1e-5):
    # P(|a||b| < sqrt t) with |a|,|b| ~ U(0,1)
    r = math.sqrt(tt)
    P = r + integrate.quad(lambda x: r / x, r, 1)[0]
    check(f"(ab)^2 volume formula at t={tt}", abs(P - float(Vab.subs(t, tt))) < 1e-12)

# ------------------------------------------------------------------ §1 Laplace-transform identity and asymptotics
# Zbar_n = n int_0^inf e^{-nt} V(t) dt; check on (ab)^2 at n = 50
def Zab(nn):  # direct 2-d integral reduced to 1-d: int_0^1 int_0^1 e^{-n a^2 b^2} db da
    f = lambda aa: math.sqrt(math.pi) * special.erf(aa * math.sqrt(nn)) / (2 * aa * math.sqrt(nn)) if aa > 0 else 1.0
    pts = [1 / math.sqrt(nn)] if nn > 1 else None
    return integrate.quad(f, 0, 1, limit=200, points=pts, epsabs=0, epsrel=1e-12)[0]

nn = 50.0
lap = nn * (mp.quad(lambda tt: mp.e ** (-nn * tt) * mp.sqrt(tt) * (1 - mp.log(tt) / 2), [0, 1 / nn, 1])
            + mp.quad(lambda tt: mp.e ** (-nn * tt), [1, mp.inf]))  # V = 1 for t >= 1
check("Zbar_n = n int e^{-nt} V(t) dt  ((ab)^2, n=50)", abs(lap / Zab(nn) - 1) < 1e-10)

# exact (ab)^2 expansion: Zbar_n = (sqrt(pi)/4) n^{-1/2} (log n + gamma + 2 log 2) + O(e^{-n})
check("psi(1/2) = -gamma - 2 log 2", abs(float(mp.digamma(0.5) + mp.euler + 2 * mp.log(2))) < 1e-25)
for nn in (10.0, 100.0, 1e4, 1e6):
    ex = Zab(nn)
    asym = math.sqrt(math.pi) / 4 * nn ** -0.5 * (math.log(nn) + np.euler_gamma + 2 * math.log(2))
    check(f"(ab)^2 closed form at n={nn:g}", abs(ex / asym - 1) < 1e-6, f"ratio-1 = {ex/asym-1:.1e}")

# local slope d Fbar / d log n = 1/2 - 1/(log n + gamma + 2 log 2)
for nn in (1e2, 1e4, 1e6):
    sl = 0.5 - 1 / (math.log(nn) + np.euler_gamma + 2 * math.log(2))
    num = -(math.log(Zab(nn * 1.01)) - math.log(Zab(nn / 1.01))) / (2 * math.log(1.01))
    check(f"(ab)^2 local slope at n={nn:g}", abs(sl - num) < 1e-5, f"{sl:.4f}")

# Abelian asymptotic: V ~ c t^lam (-log t)^{m-1} => Zbar ~ c Gamma(lam+1) n^-lam (log n)^{m-1}
# regular d-dim check of the constant: c Gamma(d/2+1) = phi0 (2 pi)^{d/2} det(H)^{-1/2}
for d in (1, 2, 3, 5):
    omega = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    H = np.diag(np.arange(1, d + 1, dtype=float))
    phi0 = 0.3
    c = phi0 * omega * 2 ** (d / 2) / math.sqrt(np.linalg.det(H))
    check(f"regular constant c*Gamma(d/2+1) = Laplace's, d={d}",
          abs(c * math.gamma(d / 2 + 1) - phi0 * (2 * math.pi) ** (d / 2) / math.sqrt(np.linalg.det(H))) < 1e-12)
# and numerically for H=diag(1,2), phi uniform density 1/16 on [-2,2]^2, K = (x^2 + 2 y^2)/2
nn = 1e4
Z = 1 / 16 * integrate.quad(lambda x: math.exp(-nn * x * x / 2), -2, 2)[0] * integrate.quad(lambda y: math.exp(-nn * y * y), -2, 2)[0]
check("Laplace on a 2-d quadratic", abs(Z / (1 / 16 * 2 * math.pi / nn / math.sqrt(2)) - 1) < 1e-10)

# ------------------------------------------------------------------ §2 zeta functions
zz = sp.symbols("zz")
def zeta_1d(expr_pow, lo=-1, hi=1, dens=sp.Rational(1, 2)):
    return sp.simplify(sp.integrate(dens * 2 * w ** expr_pow, (w, 0, hi), conds="none"))

check("zeta for w^2, uniform [-1,1]: 1/(2z+1)", sp.simplify(zeta_1d(2 * z) - 1 / (2 * z + 1)) == 0)
check("zeta for w^4, uniform [-1,1]: 1/(4z+1)", sp.simplify(zeta_1d(4 * z) - 1 / (4 * z + 1)) == 0)
zab = sp.simplify(sp.Rational(1, 4) * 4 * sp.integrate(a ** (2 * z), (a, 0, 1), conds="none") * sp.integrate(b ** (2 * z), (b, 0, 1), conds="none"))
check("zeta for (ab)^2, uniform [-1,1]^2: (2z+1)^-2", sp.simplify(zab - (2 * z + 1) ** -2) == 0)
k1, k2 = sp.symbols("k1 k2", positive=True)
zmono = sp.simplify(sp.integrate(a ** (2 * k1 * z), (a, 0, 1), conds="none") * sp.integrate(b ** (2 * k2 * z), (b, 0, 1), conds="none"))
check("zeta for a^(2k1) b^(2k2): 1/((2k1 z+1)(2k2 z+1))", sp.simplify(zmono - 1 / ((2 * k1 * z + 1) * (2 * k2 * z + 1))) == 0)
# zeta = Mellin transform of v: int_0^1 t^z (1/4) t^{-1/2}(-log t) dt
# substitute t = e^{-u} so the endpoint singularity becomes a decaying exponential
for zv in (-0.3, 0.0, 0.7):
    lhs = mp.quad(lambda u: mp.e ** (-(zv + 0.5) * u) * u / 4, [0, mp.inf])
    check(f"int t^z v(t) dt = (2z+1)^-2 for (ab)^2, z={zv}", abs(lhs / (2 * zv + 1) ** -2 - 1) < 1e-15)
# dictionary: int_0^1 t^{z+lam-1} (-log t)^{m-1}/(m-1)! dt = (z+lam)^-m
for m in (1, 2, 3):
    for zv, lv in ((-0.2, 0.5), (0.3, 0.25)):
        lhs = mp.quad(lambda u: mp.e ** (-(zv + lv) * u) * u ** (m - 1) / mp.factorial(m - 1), [0, mp.inf])
        check(f"dictionary: t^(lam-1)(-log t)^{m-1}/{m-1}! <-> (z+lam)^-{m}, z={zv}, lam={lv}",
              abs(lhs / (zv + lv) ** -m - 1) < 1e-15)
# int_0^inf e^{-nt} t^{lam-1} dt = Gamma(lam) n^-lam ; with (-log t): Gamma(lam) n^-lam (log n - psi(lam))
for lv in (0.5, 0.25, 1.0):
    for nn in (10.0, 1000.0):
        # substitute t = x^(1/lam): t^(lam-1) dt = dx / lam
        f1 = lambda x: mp.e ** (-nn * x ** (1 / lv)) / lv
        f2 = lambda x: mp.e ** (-nn * x ** (1 / lv)) * (-mp.log(x) / lv) / lv
        v1 = mp.quad(f1, [0, nn ** -lv, 1, mp.inf])
        v2 = mp.quad(f2, [0, nn ** -lv, 1, mp.inf])
        e1 = abs(v1 / (mp.gamma(lv) * nn ** -lv) - 1)
        e2 = abs(v2 / (mp.gamma(lv) * nn ** -lv * (mp.log(nn) - mp.digamma(lv))) - 1)
        check(f"Laplace of t^(lam-1)[(-log t)], lam={lv}, n={nn:g}", e1 < 1e-12 and e2 < 1e-12, f"{float(max(e1, e2)):.0e}")

# Mellin transform of Zbar_n in n: int_0^inf n^{-z-1} Zbar_n dn = Gamma(-z) zeta(z), -lam<Re z<0
for zv in (-0.1, -0.3):
    Zw2 = lambda nn: math.sqrt(math.pi) * special.erf(math.sqrt(nn)) / (2 * math.sqrt(nn))
    g = lambda u, Zf: math.exp(-zv * u) * Zf(math.exp(u))   # n = e^u, dn/n = du
    lhs = integrate.quad(g, -500, 120, args=(Zw2,), limit=1000, epsabs=0, epsrel=1e-10, points=[0])[0]
    rhs = math.gamma(-zv) / (2 * zv + 1)
    check(f"Mellin in n: int n^(-z-1) Zbar dn = Gamma(-z) zeta(z), w^2, z={zv}", abs(lhs / rhs - 1) < 1e-6, f"{lhs/rhs-1:.0e}")
    Zfull = lambda nn: Zab(nn) if nn > 1e-12 else 1.0
    lhs = integrate.quad(g, -500, 120, args=(Zfull,), limit=1000, epsabs=0, epsrel=1e-10, points=[0])[0]
    rhs = math.gamma(-zv) / (2 * zv + 1) ** 2
    check(f"Mellin in n for (ab)^2, z={zv}", abs(lhs / rhs - 1) < 1e-5, f"{lhs/rhs-1:.0e}")

# Gaussian prior, K = w^2: zeta = 2^z Gamma(z+1/2)/sqrt(pi); Zbar = (1+2n)^-1/2; residues <-> terms
zg = sp.simplify(sp.integrate(2 * (w ** 2) ** z * sp.exp(-w ** 2 / 2) / sp.sqrt(2 * sp.pi), (w, 0, sp.oo), conds="none"))
check("Gaussian prior: zeta(z) = 2^z Gamma(z+1/2)/sqrt(pi)", sp.simplify(zg - 2 ** z * sp.gamma(z + sp.Rational(1, 2)) / sp.sqrt(sp.pi)) == 0)
Zg = sp.integrate(2 * sp.exp(-n * w ** 2) * sp.exp(-w ** 2 / 2) / sp.sqrt(2 * sp.pi), (w, 0, sp.oo))
check("Gaussian prior: Zbar_n = (1+2n)^-1/2", sp.simplify(Zg - (1 + 2 * n) ** sp.Rational(-1, 2)) == 0)
for k in range(5):
    zk = -sp.Rational(1, 2) - k
    res = sp.residue(sp.gamma(-zz) * 2 ** zz * sp.gamma(zz + sp.Rational(1, 2)) / sp.sqrt(sp.pi) * n ** zz, zz, zk)
    term = sp.binomial(sp.Rational(-1, 2), k) * (2 * n) ** (-sp.Rational(1, 2) - k)
    check(f"Gaussian prior: residue at z=-1/2-{k} = n^(-1/2-{k}) term of (1+2n)^-1/2",
          sp.simplify(res - term) == 0, str(sp.simplify(res)))

# Cubic: K = mu^6/2, uniform prior on [-1,1]: zeta = 2^{-z}/(6z+1); pushforward prior on theta = mu^3
zc = sp.simplify(sp.Rational(1, 2) * 2 * sp.integrate((w ** 6 / 2) ** z, (w, 0, 1), conds="none"))
check("cubic, uniform prior: zeta = 2^-z/(6z+1)", sp.simplify(zc - 2 ** -z / (6 * z + 1)) == 0)
th = sp.symbols("th", positive=True)
zth = sp.simplify(2 * sp.integrate((th ** 2 / 2) ** z * sp.Rational(1, 2) * sp.Rational(1, 3) * th ** sp.Rational(-2, 3), (th, 0, 1), conds="none"))
for zv in (-0.1, 0.0, 0.5):
    lhs = 2 * mp.quad(lambda u: (mp.e ** (-2 * u) / 2) ** zv * mp.e ** (u * mp.mpf(2) / 3) / 6 * mp.e ** -u, [0, mp.inf])  # theta = e^-u
    check(f"cubic in theta coords, pushforward prior (1/6)|theta|^-2/3: same zeta, z={zv}",
          abs(lhs / (2 ** -zv / (6 * zv + 1)) - 1) < 1e-10)
mu0 = sp.symbols("mu0", positive=True)
Kc = (w ** 3 - mu0 ** 3) ** 2 / 2
check("cubic Hessian at mu0 is 9 mu0^4", sp.simplify(sp.diff(Kc, w, 2).subs(w, mu0) - 9 * mu0 ** 4) == 0)
# Gaussian prior version: zeta = 2^{2z} Gamma(3z+1/2)/sqrt(pi), first pole -1/6
zcg = sp.simplify(sp.integrate(2 * (w ** 6 / 2) ** z * sp.exp(-w ** 2 / 2) / sp.sqrt(2 * sp.pi), (w, 0, sp.oo), conds="none"))
check("cubic, Gaussian prior: zeta = 4^z Gamma(3z+1/2)/sqrt(pi)", sp.simplify(sp.gammasimp(zcg - 4 ** z * sp.gamma(3 * z + sp.Rational(1, 2)) / sp.sqrt(sp.pi))) == 0)

# ------------------------------------------------------------------ §3 two basins
# worksheet numbers: F_U = 0.3n + 20 log n, F_V = 0.5n + 2 log n
g = lambda x: 0.2 * x - 18 * math.log(x)
nstar = optimize.brentq(g, 10, 5000)
nlow = optimize.brentq(g, 1.0001, 5)
check("worksheet n* ~ 570", abs(nstar - 570) < 2, f"n* = {nstar:.1f}, spurious small root {nlow:.4f}")
slope = 0.2 - 18 / nstar
check("transition width 2/slope ~ 12", abs(2 / slope - 12) < 0.5, f"{2/slope:.1f}, relative {2/slope/nstar:.3f}")
# leading-order solution of n/log n = r: n ~ r log r
r = 90.0
it = [r * math.log(r)]
for _ in range(3):
    it.append(r * math.log(it[-1]))
check("iteration n <- r log n from r log r reaches n*", abs(it[-1] / nstar - 1) < 0.01, ", ".join(f"{x:.0f}" for x in it))

# d-dim two-basin example: B: sum w^2, A: dL + sum w^4, prior 1/2 uniform on each cube
d, dL = 8, 0.01
IB = lambda nn: math.sqrt(math.pi) * special.erf(math.sqrt(nn)) / (2 * math.sqrt(nn))
IA = lambda nn: integrate.quad(lambda x: math.exp(-nn * x ** 4), 0, 1)[0]
logodds = lambda nn: (-nn * dL + d * math.log(IA(nn))) - d * math.log(IB(nn))  # log Z_A/Z_B
nx = optimize.brentq(logodds, 100, 1e5)
pred = optimize.brentq(lambda x: x * dL - (d / 4) * math.log(x) - d * math.log(math.gamma(1.25) / (math.sqrt(math.pi) / 2)), 50, 1e5)
pred0 = optimize.brentq(lambda x: x * dL - (d / 4) * math.log(x), 50, 1e5)
check("two-basin exact n* vs asymptotic with constants", abs(nx / pred - 1) < 0.01,
      f"exact {nx:.0f}, with constants {pred:.0f}, leading terms only {pred0:.0f}")
# dFbar/dn = posterior mean of K
nn = 1500.0
Fb = lambda x: -math.log(0.5 * math.exp(-x * dL) * IA(x) ** d + 0.5 * IB(x) ** d)
h = 1e-3 * nn
dF = (Fb(nn + h) - Fb(nn - h)) / (2 * h)
# posterior mean of K: per-basin means via 1-d integrals
mA = dL + d * integrate.quad(lambda x: x ** 4 * math.exp(-nn * x ** 4), 0, 1)[0] / IA(nn)
mB = d * integrate.quad(lambda x: x ** 2 * math.exp(-nn * x ** 2), 0, 1)[0] / IB(nn)
pA = 1 / (1 + math.exp(-logodds(nn)))
check("dFbar/dn = E_post[K]", abs(dF - (pA * mA + (1 - pA) * mB)) < 1e-6, f"{dF:.6f}")


# ------------------------------------------------------------------ added after drafting
# a^2 b^4, uniform [-1,1]^2: Zbar_n = (1/2)Gamma(1/4) n^-1/4 - (1/2)Gamma(1/2) n^-1/2 + O(e^-n)
for nn in (1e2, 1e4):
    ex = integrate.quad(lambda bb: math.sqrt(math.pi) * special.erf(bb * bb * math.sqrt(nn)) / (2 * bb * bb * math.sqrt(nn)) if bb > 0 else 1.0,
                        0, 1, points=[nn ** -0.25], limit=400, epsabs=0, epsrel=1e-12)[0]
    asym = 0.5 * math.gamma(0.25) * nn ** -0.25 - 0.5 * math.gamma(0.5) * nn ** -0.5
    check(f"a^2 b^4 two-pole expansion at n={nn:g}", abs(ex / asym - 1) < 1e-8, f"{ex/asym-1:.0e}")
# w^4: Zbar_n ~ Gamma(5/4) n^-1/4
check("w^4: int_0^1 e^{-n w^4} ~ Gamma(5/4) n^-1/4", abs(integrate.quad(lambda x: math.exp(-1e4 * x ** 4), 0, 1, points=[0.1])[0] / (math.gamma(1.25) * 1e4 ** -0.25) - 1) < 1e-10)
# equipartition: E_post[n K] -> lambda (w^2: 1/2, w^4: 1/4) at large n
for p_, lamv in ((2, 0.5), (4, 0.25)):
    nn = 1e6
    num = integrate.quad(lambda x: nn * x ** p_ * math.exp(-nn * x ** p_), 0, 1, points=[nn ** (-1 / p_)])[0]
    den = integrate.quad(lambda x: math.exp(-nn * x ** p_), 0, 1, points=[nn ** (-1 / p_)])[0]
    check(f"E_post[nK] -> lambda for w^{p_}", abs(num / den - lamv) < 1e-6, f"{num/den:.6f}")
# n* ~ r(log r + log log r) is an asymptotic statement: relative error -> 0 as r grows
errs = []
for rr in (90.0, 1e4, 1e8):
    ns_ = optimize.brentq(lambda x: x / math.log(x) - rr, math.e * 1.0001, 1e12)
    errs.append(abs(rr * (math.log(rr) + math.log(math.log(rr))) / ns_ - 1))
check("n* ~ r(log r + log log r): error shrinks with r", errs[0] > errs[1] > errs[2], ", ".join(f"{e:.3f}" for e in errs))
# relative width formula vs exact in the d=8 example: 2/(dlam (log n* - 1))
check("width formula 2/(dlam(log n*-1)) for d=8 example", abs(2 / (2 * (math.log(nx) - 1)) - 0.16) < 0.01, f"{2/(2*(math.log(nx)-1)):.3f}")

# ------------------------------------------------------------------ lambda <= d/2 (spot check on examples)
for name, lamv, dd in [("w^4", 0.25, 1), ("x^2+y^4", 0.75, 2), ("(ab)^2", 0.5, 2), ("a^2 b^4", 0.25, 2)]:
    check(f"lambda <= d/2 for {name}", lamv <= dd / 2)

print(f"\n{len(FAIL)} failures" if FAIL else "\nall checks passed")
sys.exit(1 if FAIL else 0)
