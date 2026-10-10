#!/usr/bin/env python3
"""Numerical checks for WB2 1.F (information primer, rate-distortion warm-ups).

Run: python checks/wb2_new_ratedistortion_check.py   (numpy + scipy; a few seconds)
All logs natural (nats).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

rng = np.random.default_rng(0)
OK = True


def report(name, formula, numeric, rtol=1e-3, atol=1e-6):
    global OK
    good = np.allclose(formula, numeric, rtol=rtol, atol=atol)
    OK &= bool(good)
    print(f"  [{'ok' if good else 'FAIL'}] {name}: formula={np.round(np.asarray(formula, float), 6)}"
          f"  numeric={np.round(np.asarray(numeric, float), 6)}")


def Hb(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return np.nan_to_num(out)


def mi(px, enc):
    """I(X;T) for p(x) and encoder enc[x,t]."""
    pt = px @ enc
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = px[:, None] * enc * (np.log(enc) - np.log(pt)[None, :])
    return np.nansum(np.where(enc > 0, terms, 0.0))


def blahut_arimoto(px, d, beta, iters=5000, tol=1e-13):
    """Alternating minimisation of I(X;T) + beta E d. Returns encoder, q(t), D, R, F."""
    nt = d.shape[1]
    q = np.full(nt, 1.0 / nt)
    for _ in range(iters):
        logits = np.log(q)[None, :] - beta * d
        logZ = logsumexp(logits, axis=1)
        enc = np.exp(logits - logZ[:, None])
        qn = px @ enc
        if np.max(np.abs(qn - q)) < tol:
            q = qn
            break
        q = qn
    logits = np.log(np.maximum(q, 1e-300))[None, :] - beta * d
    logZ = logsumexp(logits, axis=1)
    enc = np.exp(logits - logZ[:, None])
    D = np.sum(px[:, None] * enc * d)
    R = mi(px, enc)
    F = -np.sum(px * logZ)
    return enc, q, D, R, F


print("Primer identities")
# random joint p(x,y,z) with Markov chain Y -> X -> T
nx, ny, nt = 5, 3, 4
py = rng.dirichlet(np.ones(ny))
pxy = rng.dirichlet(np.ones(nx), size=ny)       # p(x|y)
ptx = rng.dirichlet(np.ones(nt), size=nx)       # p(t|x)
joint = py[:, None, None] * pxy[:, :, None] * ptx[None, :, :]   # p(y,x,t)


def H(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


Hy, Hx, Ht = H(joint.sum((1, 2))), H(joint.sum((0, 2))), H(joint.sum((0, 1)))
Hyx, Hyt, Hxt = H(joint.sum(2)), H(joint.sum(1)), H(joint.sum(0))
Ixy, Iyt = Hy + Hx - Hyx, Hy + Ht - Hyt
pyx = joint.sum(2)
kl_form = np.sum(pyx * np.log(pyx / np.outer(pyx.sum(1), pyx.sum(0))))
report("I(X;Y) = H+H-H(X,Y) = KL(joint||product)", Ixy, kl_form)
print(f"  [{'ok' if Iyt <= Ixy + 1e-12 else 'FAIL'}] data processing: I(Y;T)={Iyt:.4f} <= I(Y;X)={Ixy:.4f}")
OK &= Iyt <= Ixy + 1e-12
# chain rule I(Y;X,T) = I(Y;X) + I(Y;T|X), with I(Y;T|X)=0
Hxt_all = H(joint)
IY_XT = Hy + Hxt - Hxt_all
report("chain rule: I(Y;X,T) = I(Y;X) under Markov chain", IY_XT, Ixy)
# Gaussian differential entropy and scaling
s = 1.7
xs = np.linspace(-12 * s, 12 * s, 200001)
f = np.exp(-xs**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
h_num = -np.trapz(f * np.log(f), xs)
report("h(N(0,s^2)) = 0.5 ln(2 pi e s^2)", 0.5 * np.log(2 * np.pi * np.e * s**2), h_num)
# I = min_q sum_x p(x) KL(p(t|x)||q) + identity
px = rng.dirichlet(np.ones(nx)); q = rng.dirichlet(np.ones(nt))
lhs = np.sum(px[:, None] * ptx * np.log(ptx / q[None, :]))
pt = px @ ptx
report("sum_x p(x) KL(p(t|x)||q) = I(X;T) + KL(p_T||q)", mi(px, ptx) + np.sum(pt * np.log(pt / q)), lhs)

print("\n1.F.1 Bernoulli source, Hamming distortion")
for p in [0.5, 0.3, 0.11]:
    px = np.array([1 - p, p])
    d = 1.0 - np.eye(2)
    for beta in [0.5, 1.5, 3.0, 6.0]:
        enc, q, D, R, F = blahut_arimoto(px, d, beta)
        if D < p - 1e-6:
            report(f"p={p}, beta={beta}: R = Hb(p)-Hb(D)", Hb(p) - Hb(D), R, rtol=1e-6)
            report(f"   beta = ln((1-D)/D)", np.log((1 - D) / D), beta, rtol=1e-6)
            report(f"   q(1) = (p-D)/(1-2D)", (p - D) / (1 - 2 * D), q[1], rtol=1e-6)
            report(f"   F(beta) = -E log Z = R + beta D", R + beta * D, F, rtol=1e-8)
        else:
            print(f"  [ok?] p={p}, beta={beta}: beta below ln((1-p)/p)={np.log((1-p)/p):.3f}: R={R:.2e}, D={D:.4f} (expect R=0, D=p)")
            OK &= (R < 1e-6) and abs(D - p) < 1e-6
# direct constrained minimisation over all 2x2 encoders for one D
p, Dt = 0.3, 0.1
px = np.array([1 - p, p])
A, B = np.meshgrid(np.linspace(0, 1, 2001), np.linspace(0, 1, 2001), indexing="ij")
P1 = (1 - p) * A + p * (1 - B)            # p(T=1); A = p(t=1|x=0), B = p(t=0|x=1)


def xlogy_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, a * np.log(a / b), 0.0)


Igrid = ((1 - p) * (xlogy_ratio(1 - A, 1 - P1) + xlogy_ratio(A, P1))
         + p * (xlogy_ratio(B, 1 - P1) + xlogy_ratio(1 - B, P1)))
feasible = (1 - p) * A + p * B <= Dt + 1e-12
best = Igrid[feasible].min()
report("brute-force grid min I over encoders with E d <= 0.1 (p=0.3)", Hb(p) - Hb(Dt), best, rtol=2e-3)

print("\n1.F.1 Gaussian source, squared error")
sig2 = 2.0
xg = np.linspace(-7, 7, 561) * np.sqrt(sig2)
px = np.exp(-xg**2 / (2 * sig2)); px /= px.sum()
tg = np.linspace(-6, 6, 401) * np.sqrt(sig2)
d = (xg[:, None] - tg[None, :])**2
for beta in [0.5, 1.0, 2.0, 5.0]:
    enc, q, D, R, F = blahut_arimoto(px, d, beta, iters=20000, tol=1e-11)
    report(f"beta={beta}: R = 0.5 ln(sig2/D)", 0.5 * np.log(sig2 / D), R, rtol=3e-3)
    report(f"   D = 1/(2 beta)", 1 / (2 * beta), D, rtol=3e-3)
    # optimal q(t) should be N(0, sig2 - D)
    var_q = np.sum(q * tg**2)
    report(f"   Var q = sig2 - D", sig2 - D, var_q, rtol=3e-3)
# forward channel of the backward test channel X = T + Z
D = 0.5
a, v = (sig2 - D) / sig2, D * (sig2 - D) / sig2
n = 2_000_000
T = rng.normal(0, np.sqrt(sig2 - D), n); X = T + rng.normal(0, np.sqrt(D), n)
slope = np.cov(X, T)[0, 1] / np.var(X)
resid = np.var(T - slope * X)
report("forward channel T|X ~ N(aX, v): a, v", [a, v], [slope, resid], rtol=5e-3)

print("\n1.F.1 Envelope: dR/dD = -beta (finite differences on Bernoulli BA curve)")
px = np.array([0.7, 0.3]); d = 1.0 - np.eye(2)
b1, b2 = 2.0, 2.001
_, _, D1, R1, _ = blahut_arimoto(px, d, b1)
_, _, D2, R2, _ = blahut_arimoto(px, d, b2)
report("dR/dD", -(b1 + b2) / 2, (R2 - R1) / (D2 - D1), rtol=1e-4)

print("\n1.F.1 gradient formula dI/dp(t|x) = p(x) log(p(t|x)/p(t))")
px = rng.dirichlet(np.ones(nx)); enc = rng.dirichlet(np.ones(nt), size=nx)
eps = 1e-6
G = np.zeros_like(enc)
for i in range(nx):
    for j in range(nt):
        e2 = enc.copy(); e2[i, j] += eps
        e1 = enc.copy(); e1[i, j] -= eps
        G[i, j] = (mi(px, e2) - mi(px, e1)) / (2 * eps)
pt = px @ enc
report("unconstrained partials", px[:, None] * np.log(enc / pt[None, :]), G, rtol=1e-5)

print("\n1.F.1 (optional) reverse water-filling")
sig = np.array([4.0, 2.0, 1.0, 0.3])
D = 1.8


def wf(sig, D):
    lo, hi = 0.0, sig.max()
    for _ in range(200):
        th = (lo + hi) / 2
        if np.minimum(th, sig).sum() > D: hi = th
        else: lo = th
    Di = np.minimum(th, sig)
    return th, Di, 0.5 * np.sum(np.log(sig / Di))


th, Di, Rwf = wf(sig, D)
print(f"  theta={th:.4f}, D_i={np.round(Di, 4)}")
# brute-force: minimise sum 0.5 ln+(sig/Di) s.t. sum Di = D, 0<Di
def obj(z):
    Dz = D * np.exp(z) / np.exp(z).sum()
    return 0.5 * np.sum(np.maximum(np.log(sig / Dz), 0))
res = min((minimize(obj, rng.normal(size=4), method="Nelder-Mead",
                    options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000)) for _ in range(20)),
          key=lambda r: r.fun)
report("water-filling R(D) vs numerical allocation", Rwf, res.fun, rtol=1e-4)
# R from the per-component test channels: 0.5 log det(Sigma_X)/det(Sigma_{X|T})
report("theta = 0.5 when sig=(4,2,1,0.3), D=1.8 (component 4 dropped)", 0.5, th)
print("\nALL OK" if OK else "\nSOME CHECKS FAILED")
