#!/usr/bin/env python3
"""Checks for every formula stated in slt3_temperature (NB3). Run from slt/: python checks/slt3_temperature_formulas.py

Each check prints PASS/FAIL with the numbers. Symbolic where sympy can do it, otherwise numerical.
"""
import numpy as np
import sympy as sp
from scipy import integrate, special

ok = True


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------- §1: lambda for K = w^4 + delta w^2
t, d, w = sp.symbols("t delta w", positive=True)
# delta = 0: V(t) = |{w^4 < t}| = 2 t^(1/4)
report("1.0 V(t)=2t^(1/4) for w^4", True, "(by inspection: w^4<t <=> |w|<t^(1/4))")
# delta > 0: V(t) = 2 w_t with w_t^4 + delta w_t^2 = t, w_t^2 = (-delta + sqrt(delta^2+4t))/2
wt2 = (-d + sp.sqrt(d**2 + 4 * t)) / 2
lead = sp.limit(2 * sp.sqrt(wt2) / sp.sqrt(t), t, 0)
report("1.0 V(t) ~ 2 sqrt(t/delta) as t->0 (lambda=1/2)", sp.simplify(lead - 2 / sp.sqrt(d)) == 0, f"lead coeff {lead}")


# ---------------------------------------------------------------- quadrature helpers
def grid(wmax, N=400001, c=8.0):
    s = np.linspace(-1, 1, N)
    x = wmax * np.sinh(c * s) / np.sinh(c)
    return x, np.gradient(x)


def logZ(K, n, logprior, wmax=10.0):
    x, dx = grid(wmax)
    a = -n * K(x) + logprior(x) + np.log(dx)
    return special.logsumexp(a)


def n_meanK(K, n, logprior, wmax=10.0):
    x, dx = grid(wmax)
    a = -n * K(x) + logprior(x) + np.log(dx)
    p = np.exp(a - a.max())
    return n * (p * K(x)).sum() / p.sum()


gauss = lambda x: -x**2 / 2

# Ex 1.2: lambda_eff = -dlogZ/dlog n = n <K>_n
for delta in [0.0, 1e-2]:
    K = lambda x, de=delta: x**4 + de * x**2
    for n in [1e2, 1e4, 1e6]:
        h = 1e-4
        fd = -(logZ(K, n * np.exp(h), gauss) - logZ(K, n * np.exp(-h), gauss)) / (2 * h)
        mk = n_meanK(K, n, gauss)
        report(f"1.2 lambda_eff identity delta={delta} n={n:.0e}", abs(fd - mk) < 1e-5, f"fd={fd:.6f} n<K>={mk:.6f}")

# Ex 1.3: flat prior, w = n^(-1/4) u: lambda_eff = 1/4 + (s/2)<u^2>_s with s = delta sqrt(n);
# integration by parts 4<u^4> + 2 s <u^2> = 1; closed form g(s) = int exp(-u^4 - s u^2) du
def moments(s):
    g = integrate.quad(lambda u: np.exp(-u**4 - s * u**2), -np.inf, np.inf)[0]
    u2 = integrate.quad(lambda u: u**2 * np.exp(-u**4 - s * u**2), -np.inf, np.inf)[0] / g
    u4 = integrate.quad(lambda u: u**4 * np.exp(-u**4 - s * u**2), -np.inf, np.inf)[0] / g
    return g, u2, u4


for s in [0.0, 0.3, 1.0, 3.0, 30.0]:
    g, u2, u4 = moments(s)
    report(f"1.3 IBP 4<u4>+2s<u2>=1 at s={s}", abs(4 * u4 + 2 * s * u2 - 1) < 1e-8, f"{4*u4+2*s*u2:.10f}")
    if s > 0:
        closed = 0.5 * np.sqrt(s) * np.exp(s**2 / 8) * special.kv(0.25, s**2 / 8)
        report(f"1.3 Bessel closed form g(s) at s={s}", abs(closed - g) / g < 1e-8, f"quad={g:.8f} closed={closed:.8f}")
# scaling form vs direct (flat prior on a wide interval) for two (n, delta) with equal n delta^2
flat = lambda x: 0 * x
for (n, delta) in [(1e4, 1e-2), (1e6, 1e-3)]:
    s = delta * np.sqrt(n)
    _, u2, _ = moments(s)
    direct = n_meanK(lambda x: x**4 + delta * x**2, n, flat, wmax=5.0)
    report(f"1.3 scaling form n={n:.0e} delta={delta}", abs(direct - (0.25 + s / 2 * u2)) < 1e-4,
           f"direct={direct:.5f} scaling={0.25+s/2*u2:.5f}")
_, u2, _ = moments(1.0)
print(f"      lambda_eff at n delta^2 = 1 (flat prior): {0.25 + 0.5*u2:.4f}")
for s in [1e-3, 1e3]:
    _, u2, _ = moments(s)
    print(f"      s={s}: lambda_eff={0.25+s/2*u2:.4f}")

# Ex 1.1: model y = w^2 x + noise, sigma^2 = 1/2, truth y = -a x + noise, x ~ N(0,1):
# K(w) = w^4 + 2 a w^2; K_n(w) = m2 w^4 + delta_n w^2, delta_n = -2 mean(xy), sd(delta_n - delta m2) = sqrt(2/n)
a_, x_ = sp.symbols("a x", real=True)
Ksym = sp.expand(((w**2 + a_)**2 - a_**2))  # E[(w^2 x + a x)^2 - (a x)^2]/(2 sigma^2) with E x^2 = 1, 2 sigma^2 = 1
report("1.1 K = w^4 + 2 a w^2", sp.simplify(Ksym - (w**4 + 2 * a_ * w**2)) == 0, str(Ksym))
rng = np.random.default_rng(0)
n, a, reps = 1000, 0.0, 4000
x = rng.standard_normal((reps, n))
y = -a * x + np.sqrt(0.5) * rng.standard_normal((reps, n))
dn = -2 * np.mean(x * y, 1)
report("1.1 sd(delta_n) = sqrt(2/n) at a=0", abs(dn.std() / np.sqrt(2 / n) - 1) < 0.05,
       f"sd={dn.std():.5f} sqrt(2/n)={np.sqrt(2/n):.5f}")

# Ex 1.2 (limit): multiplicity correction lambda_eff ~ lambda - (m-1)/log n, K = a^2 b^2, uniform prior on [-1,1]^2
# Z(n) = int int exp(-n a^2 b^2) = int_{-1}^{1} sqrt(pi/n)/|a| erf(sqrt(n)|a|) da
def logZab(n):
    f = lambda a: 2 * (np.sqrt(np.pi / n) / (2 * a) * special.erf(np.sqrt(n) * a)) if a > 0 else 2.0
    return np.log(2 * integrate.quad(f, 0, 1, points=[1 / np.sqrt(n)], limit=200)[0])


# Closed form: Z(n) = 2 sqrt(pi/n) G(sqrt n), G(X) = int_0^X erf(v)/v dv, so
# lambda_eff = 1/2 - erf(sqrt n)/(2 G(sqrt n)) = 1/2 - 1/(log n + 2C) + o(1/log n), C = lim G(X) - log X.
G = lambda X: integrate.quad(lambda v: special.erf(v) / v, 0, 1)[0] + integrate.quad(lambda v: special.erf(v) / v, 1, X, limit=500)[0]
C = G(1e6) - np.log(1e6)
for n in [1e4, 1e8]:
    h = 1e-3
    fd = -(logZab(n * np.exp(h)) - logZab(n * np.exp(-h))) / (2 * h)
    cf = 0.5 - special.erf(np.sqrt(n)) / (2 * G(np.sqrt(n)))
    report(f"1.2 a^2b^2 closed form lambda_eff at n={n:.0e}", abs(fd - cf) < 1e-3, f"quad={fd:.4f} closed={cf:.4f}")
for n in [1e4, 1e8, 1e16, 1e64]:
    cf = 0.5 - special.erf(np.sqrt(n)) / (2 * G(np.sqrt(n))) if n < 1e13 else 0.5 - 1 / (np.log(n) + 2 * C)
    print(f"      a^2b^2: n={n:.0e} lambda_eff={cf:.4f}  1/2-1/log n={0.5-1/np.log(n):.4f}  (log n)*(1/2-lambda_eff)={np.log(n)*(0.5-cf):.3f}")
report("1.2 m=2: (log n)(lambda - lambda_eff) -> m-1 = 1", abs(np.log(1e64) * (1 / (np.log(1e64) + 2 * C)) - 1) < 0.02, f"C={C:.4f}")

# ---------------------------------------------------------------- §3
# Ex 3.1 WBIC: F(beta) = n beta L0 + lam log(n beta);  dF/dbeta = n L0 + lam/beta; equals F(1) at beta = 1/log n
beta, n_, L0, lam = sp.symbols("beta n L_0 lambda", positive=True)
Fb = n_ * beta * L0 + lam * sp.log(n_ * beta)
dF = sp.diff(Fb, beta)
sol = sp.solve(sp.Eq(dF, Fb.subs(beta, 1)), beta)
report("3.1 WBIC: E_beta[nL_n] = F_n at beta = 1/log n", sp.simplify(sol[0] - 1 / sp.log(n_)) == 0, str(sol))
report("3.1 estimator: beta (dF/dbeta - n L0) = lambda", sp.simplify(beta * (dF - n_ * L0) - lam) == 0)

# Ex 3.2 regular case, exact: K_n(w) = 1/2 (w-wh)^T J (w-wh) - 1/2 wh^T J wh, posterior N(wh, (n beta J)^-1)
# => lambda_hat = d/2 - (n beta/2) wh^T J wh;  E = (d/2)(1 - beta) when n wh^T J wh ~ chi^2_d
J = np.diag([1.0, 3.0, 0.5]); dd = 3; n, b = 5000, 0.2
wh = rng.multivariate_normal(np.zeros(dd), np.linalg.inv(n * J))
S = rng.multivariate_normal(wh, np.linalg.inv(n * b * J), size=400000)
Kn = 0.5 * np.einsum("si,ij,sj->s", S - wh, J, S - wh) - 0.5 * wh @ J @ wh
mc = n * b * Kn.mean()
pred = dd / 2 - n * b / 2 * wh @ J @ wh
report("3.2 regular-case lambda_hat formula (MC)", abs(mc - pred) < 0.02, f"MC={mc:.4f} formula={pred:.4f}")
whs = rng.multivariate_normal(np.zeros(dd), np.linalg.inv(n * J), size=200000)
avg = np.mean(dd / 2 - n * b / 2 * np.einsum("si,ij,sj->s", whs, J, whs))
report("3.2 E lambda_hat = (d/2)(1-beta)", abs(avg - dd / 2 * (1 - b)) < 0.01, f"{avg:.4f} vs {dd/2*(1-b):.4f}")

# Ex 3.3 localisation, 1-D quadratic K = h w^2/2: n beta <K> = 1/(2(1 + gamma/(n beta h)))
hh, g_, nb = sp.symbols("h gamma N", positive=True)
prec = nb * hh + g_
val = sp.simplify(nb * hh / 2 / prec)
report("3.3 gamma cutoff", sp.simplify(val - sp.Rational(1, 2) / (1 + g_ / (nb * hh))) == 0, str(val))

# ---------------------------------------------------------------- DLN: Aoyagi & Watanabe (2005), as stated in the worksheet
def aoyagi(M, N, H, r):
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        s = -(H + r)**2 - M**2 - N**2 + 2 * (H + r) * M + 2 * (H + r) * N + 2 * M * N
        return (sp.Rational(s, 8), 1) if (M + H + N + r) % 2 == 0 else (sp.Rational(s + 1, 8), 2)
    if M + H < N + r:
        return sp.Rational(H * M - H * r + N * r, 2), 1
    if N + H < M + r:
        return sp.Rational(H * N - H * r + M * r, 2), 1
    return sp.Rational(M * N, 2), 1


# Drton & Plummer (2017, arXiv:1309.0911) Table 1: N=5 responses, M=3 covariates, model rank i (=H), true rank j (=r);
# multiplicity 1 except (i,j)=(3,0) where it is 2.
table = {(1, 0): sp.Rational(3, 2), (1, 1): sp.Rational(7, 2), (2, 0): 3, (2, 1): sp.Rational(9, 2), (2, 2): 6,
         (3, 0): sp.Rational(9, 2), (3, 1): sp.Rational(11, 2), (3, 2): sp.Rational(13, 2), (3, 3): sp.Rational(15, 2)}
allok = True
for (i, j), v in table.items():
    lamv, mult = aoyagi(3, 5, i, j)
    allok &= (lamv == v) and (mult == (2 if (i, j) == (3, 0) else 1))
report("DLN formula reproduces Drton-Plummer Table 1 (N=5, M=3)", allok)
report("DLN M=N=H=1, r=0: lambda=1/2, m=2 (K ~ (ab)^2)", aoyagi(1, 1, 1, 0) == (sp.Rational(1, 2), 2))
report("DLN M=N=H=1, r=1: lambda=1/2, m=1", aoyagi(1, 1, 1, 1) == (sp.Rational(1, 2), 1))
report("DLN M=N=2, H=1, r=0: lambda=1, m=2 (|a|^2|b|^2)", aoyagi(2, 2, 1, 0) == (1, 2))
print("      NB3 DLN (M=3 in, N=5 out, H=3):", [aoyagi(3, 5, 3, r) for r in range(4)], " d/2 =", 3 * 8 / 2)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
