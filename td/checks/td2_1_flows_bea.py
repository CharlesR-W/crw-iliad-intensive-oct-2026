"""Symbolic checks for td2_1_flows (backward error analysis, momentum).

Run: python td/checks/td2_1_flows_bea.py   (well under a minute)

1. The general BEA recipe f2 = d2 - f'f/2,
   f3 = d3 - (f''(f,f) + f'f'f)/6 - (f'f2 + f2'f)/2, tested exactly on random
   polynomial vector fields in 2D (identity of polynomials in x, y, h).
2. GD: f2 = -H g/2 = -grad(|g|^2/4) (IGR); f3 = -H^2 g/3 - T(g,g)/12; f3 has
   nonzero curl for a concrete loss, so it is not a gradient field.
3. Quadratic: log(1 - eta lam)/eta series.
4. Momentum family u_{n+1} = u_n + beta (u_n - u_{n-1}) - h grad L(u_n + a (u_n - u_{n-1}))
   (Kovachki-Stuart's eq. (7) with their lambda -> beta; a=0 heavy ball, a=beta Nesterov).
   Slow-manifold step D(u) = h D1 + h^2 D2, then BEA on u -> u + D(u).
   Result: du/dt = -(1/(1-beta)) grad[L + (h/2) c |grad L|^2],
   c = lbar (lbar - a - 1/2), lbar = 1/(1-beta). Heavy ball: (h/2)c = h(1+beta)/(4(1-beta)^2)
   (Ghosh et al. Thm 4.1). Kovachki-Stuart print c = lbar(lbar - a + 1/2) (their eq. 19);
   the printed value fails the beta=0 check (GD needs c = 1/2).
5. The heavy-ball modified flow satisfies the heavy-ball recursion to O(h^2) on a concrete
   2D loss (independent of the slow-manifold derivation in 4).
6. Kovachki-Stuart's intermediate step: for du/dt = -lbar grad Phi + O(h),
   d2u/dt2 = +lbar^2 H grad Phi = +(lbar^2/2) grad|grad Phi|^2 (they print a minus sign),
   and redoing their matching with the correct sign gives c = lbar(lbar - a - 1/2).
"""
import random

import sympy as sp

h = sp.symbols('h')
x, y = sp.symbols('x y')
Y = sp.Matrix([x, y])


def trunc(v, order):
    return v.applyfunc(lambda e: sum(sp.expand(e).coeff(h, k) * h**k for k in range(order + 1)))


def jac(v):
    return v.jacobian(Y)


def flow_series(F, order):
    """Taylor series in h (to h^order) of the time-h flow of ydot = F(y): sum h^k/k! D^k y."""
    cur, out = Y, Y
    for k in range(1, order + 1):
        cur = trunc(jac(cur) * F, order - k)
        out = out + h**k / sp.factorial(k) * cur
    return trunc(out, order)


def second(fv, u, w):
    return sp.Matrix([sum(sp.diff(fv[i], Y[j], Y[k]) * u[j] * w[k]
                          for j in range(2) for k in range(2)) for i in range(2)])


def rpoly(deg=2):
    return sum(sp.Rational(random.randint(-3, 3), random.randint(1, 3)) * x**i * y**j
               for i in range(deg + 1) for j in range(deg + 1 - i))


def iszero(v):
    return v.applyfunc(sp.expand) == sp.zeros(*v.shape)


random.seed(0)
# ---- 1. general recipe
for trial in range(2):
    f = sp.Matrix([rpoly(), rpoly()])
    d2 = sp.Matrix([rpoly(), rpoly()])
    d3 = sp.Matrix([rpoly(), rpoly()])
    Jf = jac(f)
    F2 = d2 - Jf * f / 2
    F3 = d3 - (second(f, f, f) + Jf * Jf * f) / 6 - (Jf * F2 + jac(F2) * f) / 2
    lhs = flow_series(f + h * F2 + h**2 * F3, 3)
    assert iszero(lhs - (Y + h * f + h**2 * d2 + h**3 * d3))
print("1. general recipe f2, f3: exact on random polynomial fields")

# ---- 2. GD
for Lc in [rpoly(3), (x * y - 1)**2 / 2 + x**4 / 4, (x * y - 1)**2 / 2, x**2 * y / 2]:
    g = sp.Matrix([sp.diff(Lc, x), sp.diff(Lc, y)])
    H = jac(g)
    f = -g
    Jf = jac(f)
    F2 = -Jf * f / 2
    igr = Lc + h / 4 * (g.T * g)[0]
    assert iszero(-sp.Matrix([sp.diff(igr, x), sp.diff(igr, y)]) - (f + h * F2))
    F3 = -(second(f, f, f) + Jf * Jf * f) / 6 - (Jf * F2 + jac(F2) * f) / 2
    assert iszero(F3 - (-H * H * g / 3 - second(g, g, g) / 12))
    assert iszero(flow_series(f + h * F2 + h**2 * F3, 3) - (Y + h * f))
    curl = sp.expand(sp.diff(F3[1], x) - sp.diff(F3[0], y))
    assert curl != 0
    if Lc == x**2 * y / 2:   # worked by hand in Exercise 1.2.b.3
        assert iszero(F3 - sp.Matrix([-x * y**3 / 3 - sp.Rational(7, 12) * x**3 * y,
                                      -sp.Rational(5, 12) * x**2 * y**2 - x**4 / 6]))
        assert sp.expand(curl + x * (x**2 - 2 * y**2) / 12) == 0
print("2. GD: f2 = -Hg/2 = -grad(|g|^2/4); f3 = -H^2 g/3 - T(g,g)/12; curl f3 != 0")

# ---- 3. quadratic
lam, eta = sp.symbols('lambda eta', positive=True)
print("3. log(1-eta lam)/eta =", sp.series(sp.log(1 - eta * lam) / eta, eta, 0, 4))

# ---- 4. momentum family, generic 1D loss, exact
beta, a, z = sp.symbols('beta a z')
P = sp.Function('P')(z)
gp = sp.diff(P, z)
u2 = sp.symbols('u2')
D1 = -gp / (1 - beta)
D = h * D1 + h**2 * u2
# u_{n-1} = u_n - D(u_{n-1}), so D(u_{n-1}) = D(u_n) - D'(u_n) D(u_n) + O(h^3)
Dprev = D - h**2 * sp.diff(D1, z) * D1
grad_shift = gp + sp.diff(gp, z) * (a * Dprev)   # gradient at u_n + a (u_n - u_{n-1})
eqn = sp.expand(D - (beta * Dprev - h * grad_shift))
assert sp.simplify(eqn.coeff(h, 1)) == 0
D2 = sp.solve(eqn.coeff(h, 2), u2)[0]
F2 = sp.simplify(D2 - sp.diff(D1, z) * D1 / 2)
coef = sp.factor(sp.simplify(F2 / (sp.diff(P, z, 2) * gp)))
c = sp.factor(-coef * (1 - beta))          # F2 = -(1/(1-beta)) c H g
lb = sp.symbols('lbar')
c_lb = sp.simplify(c.subs(beta, 1 - 1 / lb))
print("4. c(beta, a) =", c, "; in lbar:", sp.expand(c_lb))
assert sp.simplify(c_lb - lb * (lb - a - sp.Rational(1, 2))) == 0
assert sp.simplify(c_lb - lb * (lb - a + sp.Rational(1, 2))) != 0
assert sp.simplify(c.subs(a, 0) / 2 - (1 + beta) / (4 * (1 - beta)**2)) == 0
assert c.subs({beta: 0, a: 0}) == sp.Rational(1, 2)
print("   heavy ball: (h/2)c = h(1+beta)/(4(1-beta)^2) [Ghosh et al.]: OK")
print("   Nesterov (a=beta): (h/2)c =", sp.factor(c.subs(a, beta) / 2), "* h")
print("   KS printed lbar(lbar-a+1/2): at beta=0 gives c=3/2, GD needs 1/2 -> sign error")

# ---- 5. concrete 2D heavy-ball check
for b in [sp.Rational(1, 3), sp.Rational(9, 10)]:
    Lc = (x * y - 1)**2 / 2 + x**4 / 4
    g = sp.Matrix([sp.diff(Lc, x), sp.diff(Lc, y)])
    H = jac(g)
    lbv = 1 / (1 - b)
    F = -lbv * g - h * lbv**2 * (lbv - sp.Rational(1, 2)) * H * g   # -lbar grad[L + (h/2) c abs(g)^2]
    fwd = flow_series(F, 3)      # u_{n+1} from u_n
    bwd = flow_series(-F, 3)     # u_{n-1} from u_n
    resid = trunc((fwd - Y) - b * (Y - bwd) + h * g, 2)
    assert iszero(resid)
    # negative control: Kovachki-Stuart's printed constant (+1/2) does not satisfy the recursion
    Fks = -lbv * g - h * lbv**2 * (lbv + sp.Rational(1, 2)) * H * g
    resid_ks = trunc((flow_series(Fks, 3) - Y) - b * (Y - flow_series(-Fks, 3)) + h * g, 2)
    assert not iszero(resid_ks)
print("5. heavy-ball recursion satisfied to O(h^2) by the modified flow (beta=1/3, 9/10)")

# ---- 6. KS intermediate sign
Phi = sp.Function('Phi')(z)
lbs, cs = sp.symbols('lbar c')
udot = -lbs * sp.diff(Phi, z)               # leading order
uddot = sp.diff(udot, z) * udot             # chain rule: d/dt F(u) = F'(u) udot
assert sp.simplify(uddot - lbs**2 * sp.diff(Phi, z, 2) * sp.diff(Phi, z)) == 0
# match u_n = u_{n-1} + h udot_n - (h^2/2) uddot_n with KS (18b):
# u_n = u_{n-1} - h lbar grad[Phi + h lbar (lbar - a) |grad Phi|^2 / 2]
G = sp.diff(Phi, z, 2) * sp.diff(Phi, z)   # = grad |grad Phi|^2 / 2
lhs_h2 = -lbs * cs * G - sp.Rational(1, 2) * lbs**2 * G   # h^2 coefficient with modified flow
rhs_h2 = -lbs**2 * (lbs - a) * G
sol = sp.solve(sp.simplify((lhs_h2 - rhs_h2) / G), cs)[0]
assert sp.simplify(sol - lbs * (lbs - a - sp.Rational(1, 2))) == 0
print("6. KS: correct sign of u'' gives c = lbar(lbar - a - 1/2); their printed minus sign gives +1/2")

# ---- 7. slow dynamics of the damped ODE (Exercise 1.5.b.3) reproduce the heavy-ball modified flow
eta_, b_ = sp.symbols('eta beta', positive=True)
Lz = sp.Function('L')(z)
gz, Hz = sp.diff(Lz, z), sp.diff(Lz, z, 2)
v0 = -gz / (1 - b_)
acc = sp.diff(v0, z) * v0                         # theta'' at leading order
vel = -gz / (1 - b_) - eta_ * (1 + b_) / (2 * (1 - b_)) * acc
target = -gz / (1 - b_) - eta_ * (1 + b_) / (2 * (1 - b_)**3) * Hz * gz
assert sp.simplify(vel - target) == 0
print("7. damped ODE slow dynamics = heavy-ball modified flow to O(eta)")
