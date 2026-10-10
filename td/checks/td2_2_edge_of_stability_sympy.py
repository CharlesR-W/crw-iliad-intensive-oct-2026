"""Symbolic checks for td2_2_edge_of_stability (run: python td/checks/td2_2_edge_of_stability_sympy.py).

1. Catapult: exact one-step maps for f = v.u/sqrt(n), x=1, target 0, L = f^2/2.
2. Self-stabilization toy: period-2 orbit, central-flow sigma^2 = period-2 x^2,
   two-step Jacobian at the orbit.
3. Heavy-ball / EMA-momentum thresholds on a quadratic.
4. Central-flow sigma^2 from dS/dt = 0, and the projection identity.
5. Batch-step Rayleigh-quotient criterion.
"""
import sympy as sp

ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or c
allok = True

# ---------------------------------------------------------------- 1. catapult
n = 3  # check with explicit vectors of symbolic entries, then general n via identities
eta = sp.symbols("eta", positive=True)
N = sp.symbols("n", positive=True)
us = sp.symbols("u0:%d" % n); vs = sp.symbols("v0:%d" % n)
u = sp.Matrix(us); v = sp.Matrix(vs)
f = (v.T * u)[0] / sp.sqrt(N)
lam = ((u.T * u)[0] + (v.T * v)[0]) / N
L = f**2 / 2
gu = sp.Matrix([sp.diff(L, s) for s in us]); gv = sp.Matrix([sp.diff(L, s) for s in vs])
u1 = u - eta * gu; v1 = v - eta * gv
f1 = (v1.T * u1)[0] / sp.sqrt(N)
lam1 = ((u1.T * u1)[0] + (v1.T * v1)[0]) / N
allok &= ok(sp.simplify(f1 - f * (1 - eta * lam + eta**2 * f**2 / N)) == 0, "catapult f map")
allok &= ok(sp.simplify(lam1 - (lam + eta * f**2 / N * (eta * lam - 4))) == 0, "catapult lambda map")
# Hessian at f=0 equals grad f grad f^T, top eigenvalue |grad f|^2 = lambda
gradf = sp.Matrix([sp.diff(f, s) for s in us + vs])
allok &= ok(sp.simplify((gradf.T * gradf)[0] - lam) == 0, "|grad f|^2 = lambda (NTK = sharpness at f=0)")

# ---------------------------------------------------------------- 2. toy
x, y = sp.symbols("x y", real=True)
S = sp.Function("S"); ell = sp.Function("ell")
Lt = S(y) * x**2 / 2 + ell(y)
xs = x - eta * sp.diff(Lt, x); ys = y - eta * sp.diff(Lt, y)
allok &= ok(sp.simplify(xs - (1 - eta * S(y)) * x) == 0, "toy x-map")
allok &= ok(sp.simplify(ys - (y - eta * (sp.Rational(1, 2) * S(y).diff(y) * x**2 + ell(y).diff(y)))) == 0, "toy y-map")
# period-2: x -> -x needs eta S = 2; y fixed needs x^2 = -2 ell'/S'
x2star = -2 * ell(y).diff(y) / S(y).diff(y)
allok &= ok(sp.simplify((ys - y).subs(x**2, x2star).subs(x, sp.sqrt(x2star))) == 0, "toy period-2 x^2")
# central-flow sigma^2 at (0,y): grad L = (0, ell'), grad S = (0, S')
gL = sp.Matrix([0, ell(y).diff(y)]); gS = sp.Matrix([0, S(y).diff(y)])
sigma2 = -2 * (gL.T * gS)[0] / (gS.T * gS)[0]
allok &= ok(sp.simplify(sigma2 - x2star) == 0, "toy central-flow sigma^2 = period-2 x^2")
# two-step Jacobian at the orbit, linear S and ell'' = c
a, b, c = sp.symbols("a b c", positive=True)
Sl = lambda yy: a * yy; ellp = lambda yy: -b + c * yy
def step(X, Y):
    return ((1 - eta * Sl(Y)) * X, Y - eta * (a * X**2 / 2 + ellp(Y)))
ystar = 2 / (eta * a); xstar = sp.sqrt(2 * (b - c * ystar) / a)
X, Y = sp.symbols("X Y")
m1 = sp.Matrix(step(X, Y)); J = m1.jacobian([X, Y])
J1 = J.subs({X: xstar, Y: ystar}); J2 = J.subs({X: -xstar, Y: ystar})
JJ = sp.simplify(J2 * J1)
detJJ = sp.factor(sp.simplify(JJ.det()))
allok &= ok(sp.simplify(detJJ - (1 - eta * c + eta**2 * a**2 * xstar**2)**2) == 0,
            "two-step det = (1 - eta*ell'' + eta^2 S'^2 x*^2)^2")
print("   two-step Jacobian trace:", sp.simplify(JJ.trace()))
# undamped linear case: eigenvalues 1-k^2 +- 2ik, |mu| = 1 + k^2 with k = eta a x*
k = sp.symbols("k", positive=True)
JJ0 = sp.Matrix([[-1, k], [k, 1]]) * sp.Matrix([[-1, -k], [-k, 1]])
ev = list(JJ0.eigenvals().keys())
allok &= ok(all(sp.simplify(sp.Abs(e)**2 - (1 + k**2)**2) == 0 for e in ev), "undamped |mu| = 1 + k^2")

# ---------------------------------------------------------------- 3. momentum thresholds
lam_, beta = sp.symbols("lambda beta", positive=True)
z = sp.symbols("z")
# heavy ball x_{k+1} = x_k - eta lam x_k + beta (x_k - x_{k-1})
p = z**2 - (1 + beta - eta * lam_) * z + beta
# Jury conditions for z^2 + a1 z + a0: |a0|<1, 1+a1+a0>0, 1-a1+a0>0
a1 = -(1 + beta - eta * lam_); a0 = beta
cond_minus = sp.simplify(1 - a1 + a0)  # = 2 + 2 beta - eta lam
allok &= ok(sp.simplify(cond_minus - (2 + 2 * beta - eta * lam_)) == 0, "heavy ball: p(-1) > 0 <=> eta lam < 2(1+beta)")
allok &= ok(sp.simplify(1 + a1 + a0 - eta * lam_) == 0, "heavy ball: p(1) > 0 <=> eta lam > 0")
# EMA momentum m = beta m + (1-beta) g ; x -= eta m  ==  heavy ball with eta -> eta(1-beta)
xm1, x0, m0, g0 = sp.symbols("x_m1 x_0 m_0 g_0")
m_prev = (xm1 - x0) / eta  # since x0 = x_m1 - eta m_prev
m_new = beta * m_prev + (1 - beta) * g0
x_new = x0 - eta * m_new
hb = x0 - eta * (1 - beta) * g0 + beta * (x0 - xm1)
allok &= ok(sp.simplify(x_new - hb) == 0, "EMA momentum = heavy ball with eta(1-beta)")
thr = 2 * (1 + beta) / (1 - beta)
allok &= ok(thr.subs(beta, sp.Rational(9, 10)) == 38, "38/eta at beta1 = 0.9")

# ---------------------------------------------------------------- 4. central flow sigma^2
d = 4
gl = sp.Matrix(sp.symbols("gL0:%d" % d)); gs = sp.Matrix(sp.symbols("gS0:%d" % d))
s2 = sp.symbols("sigma2")
# dS/dt = <grad S, w_dot> with w_dot = -(grad L + s2/2 grad S); set to 0
sol = sp.solve(sp.Eq((gs.T * (-(gl + s2 / 2 * gs)))[0], 0), s2)[0]
allok &= ok(sp.simplify(sol - (-2 * (gl.T * gs)[0] / (gs.T * gs)[0])) == 0, "sigma^2 = -2<gL,gS>/|gS|^2")
wdot = -(gl + sol / 2 * gs)
proj = -(gl - (gl.T * gs)[0] / (gs.T * gs)[0] * gs)
allok &= ok(sp.simplify(wdot - proj) == sp.zeros(d, 1), "central flow = projected gradient flow")
# time-averaged loss: L + 1/2 sigma^2 S at S = 2/eta
allok &= ok(sp.simplify(s2 / 2 * (2 / eta) - s2 / eta) == 0, "L + sigma^2 S/2 = L + sigma^2/eta at S = 2/eta")

# ---------------------------------------------------------------- 5. batch Rayleigh quotient
gB2, qB = sp.symbols("gB2 qB", positive=True)  # |g_B|^2 and g_B^T H_B g_B
dL = -eta * gB2 + eta**2 / 2 * qB
allok &= ok(sp.simplify(sp.solve(sp.Eq(dL, 0), qB)[0] / gB2 - 2 / eta) == 0, "batch step decreases L_B iff qB/|gB|^2 < 2/eta")


# ---------------------------------------------------------------- 6. exact modified flow on a quadratic (Ex 2.1)
z_ = sp.symbols("z")
a_series = sp.series(sp.log(1 - eta * lam_) / eta, eta, 0, 3).removeO()
allok &= ok(sp.simplify(a_series - (-lam_ - eta * lam_**2 / 2 - eta**2 * lam_**3 / 3)) == 0, "log(1-eta lam)/eta = -lam - eta lam^2/2 - ...")

# ---------------------------------------------------------------- 7. Rod Flow on a 1-D quadratic (Ex 2.6), t = k eta
S_, wb, dl = sp.symbols("S wbar delta", real=True)
gp, gm = S_ * (wb + dl), S_ * (wb - dl)
dSigma = eta / 4 * (gp**2 + gm**2) - 2 / eta * dl**2
allok &= ok(sp.simplify(dSigma.subs(wb, 0) - (eta * S_**2 / 2 - 2 / eta) * dl**2) == 0, "Rod Flow dSigma/dt at wbar=0")
dwb = -(gp + gm) / 2 - eta / 8 * (2 * S_) * (gp + gm)
allok &= ok(sp.simplify(dwb.subs(dl, 0) - (-S_ * wb - eta / 2 * S_**2 * wb)) == 0, "Rod Flow wbar eq at delta=0 = IGR flow")

# ---------------------------------------------------------------- 8. river valley stationary hill loss (Ex 2.8b)
gam, sig = sp.symbols("gamma sigma", positive=True)
V = sp.symbols("V")
Vst = sp.solve(sp.Eq(V, (1 - eta * gam)**2 * V + eta**2 * sig**2), V)[0]
allok &= ok(sp.simplify(gam * Vst / 2 - eta * sig**2 / (2 * (2 - eta * gam))) == 0, "stationary hill loss = eta sigma^2 / (2(2 - eta gamma))")

# ---------------------------------------------------------------- 9. Rod Flow exact Sigma identity (Ex 2.4a), numeric on a nonlinear loss
import numpy as _np
_A = _np.array([[3.0, 0.4], [0.4, 1.0]])
_grad = lambda w: _A @ w + 0.5 * _np.array([w[0] ** 3, _np.sin(w[1])])
_eta, _w = 0.3, _np.array([0.7, -0.4]); _ws = [_w]
for _ in range(6):
    _ws.append(_ws[-1] - _eta * _grad(_ws[-1]))
_err = 0.0
for k in range(4):
    wb = (_ws[k + 1] + _ws[k]) / 2; d = (_ws[k + 1] - _ws[k]) / 2; d1 = (_ws[k + 2] - _ws[k + 1]) / 2
    gp, gm = _grad(wb + d), _grad(wb - d)
    lhs = _np.outer(d1, d1) - _np.outer(d, d)
    rhs = _eta**2 / 4 * (_np.outer(gp, gp) + _np.outer(gm, gm)) - 2 * _np.outer(d, d)
    _err = max(_err, _np.abs(lhs - rhs).max(), _np.abs((_ws[k + 2] + _ws[k + 1]) / 2 - wb + _eta / 2 * (gp + gm)).max())
allok &= ok(_err < 1e-12, f"Rod Flow exact midpoint and Sigma identities on a nonlinear loss (max err {_err:.1e})")

print("ALL PASS" if allok else "SOME FAILED"); raise SystemExit(0 if allok else 1)
