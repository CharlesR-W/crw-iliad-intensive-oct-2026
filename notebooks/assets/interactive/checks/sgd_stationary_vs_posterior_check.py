"""Check for sgd_stationary_vs_posterior.html (WB2 sec. 6.3-6.4).

Model: L(theta) = 1/2 theta^T H theta, H = diag(1, kappa).
Noise: C = c R(phi) diag(1, kappa^gamma) R(phi)^T   (C = c H^gamma rotated by phi).
SDE:   d theta = -H theta dt + sqrt(eta/B) C^{1/2} dW   ->  H S + S H = (eta/B) C
SGD:   theta <- theta - eta (H theta + xi), Cov xi = C/B
       ->  S = (I - eta H) S (I - eta H)^T + eta^2 C / B
Posterior (tempered Laplace): S_post = (T/n) H^{-1}.

Figure formulas checked here (same algebra as the JS):
  * 2x2 Lyapunov solved as a 3x3 linear system (continuous and discrete)
  * closed form S = (eta/2B) C H^{-1} when C, H commute
  * excess risk 1/2 Tr(H S) = eta Tr(C) / (4B)  (continuous, any C)
  * temperature-optimised KL( N(0,S) || N(0,(T/n)H^{-1}) ):
        T* = n Tr(H S)/p,  KL* = 1/2 [ p log(Tr(HS)/p) - log det(HS) ]
Independent checks: scipy Lyapunov solvers, Euler-Maruyama simulation of
the SDE (small dt), direct simulation of discrete SGD, numeric minimisation of KL over T.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov, sqrtm
from scipy.optimize import minimize_scalar

N_DATA = 1000  # n used in the figure


def rot(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def make_HC(kappa, gamma, phi, c=1.0):
    H = np.diag([1.0, kappa])
    R = rot(phi)
    C = c * R @ np.diag([1.0, kappa ** gamma]) @ R.T
    return H, C


def solve3(M, q):
    return np.linalg.solve(M, q)


def lyap_cont_3x3(H, Q):
    """H S + S H = Q, S symmetric [[a,b],[b,d]]."""
    h11, h12, h22 = H[0, 0], H[0, 1], H[1, 1]
    M = np.array([[2 * h11, 2 * h12, 0.0],
                  [h12, h11 + h22, h12],
                  [0.0, 2 * h12, 2 * h22]])
    a, b, d = solve3(M, np.array([Q[0, 0], Q[0, 1], Q[1, 1]]))
    return np.array([[a, b], [b, d]])


def lyap_disc_3x3(A, Q):
    """S - A S A^T = Q, A symmetric."""
    a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
    M = np.eye(3) - np.array([[a11 ** 2, 2 * a11 * a12, a12 ** 2],
                              [a11 * a12, a11 * a22 + a12 ** 2, a12 * a22],
                              [a12 ** 2, 2 * a12 * a22, a22 ** 2]])
    a, b, d = solve3(M, np.array([Q[0, 0], Q[0, 1], Q[1, 1]]))
    return np.array([[a, b], [b, d]])


def kl_gauss(S0, S1):
    p = S0.shape[0]
    S1i = np.linalg.inv(S1)
    return 0.5 * (np.trace(S1i @ S0) - p + np.log(np.linalg.det(S1) / np.linalg.det(S0)))


def kl_star(H, S):
    p = H.shape[0]
    HS = H @ S
    tr = np.trace(HS)
    return 0.5 * (p * np.log(tr / p) - np.log(np.linalg.det(HS))), N_DATA * tr / p


def em_sde(H, C, etaB, dt, n_steps, n_chains, rng):
    L = np.linalg.cholesky(C)
    th = np.zeros((n_chains, 2))
    acc = np.zeros((2, 2)); cnt = 0
    burn = n_steps // 4
    sq = np.sqrt(etaB * dt)
    for t in range(n_steps):
        z = rng.standard_normal((n_chains, 2)) @ L.T
        th = th - dt * th @ H.T + sq * z
        if t >= burn and t % 5 == 0:
            acc += th.T @ th / n_chains; cnt += 1
    return acc / cnt


def sim_sgd(H, C, eta, B, n_steps, n_chains, rng):
    L = np.linalg.cholesky(C / B)
    th = np.zeros((n_chains, 2))
    acc = np.zeros((2, 2)); cnt = 0
    burn = n_steps // 4
    for t in range(n_steps):
        xi = rng.standard_normal((n_chains, 2)) @ L.T
        th = th - eta * (th @ H.T + xi)
        if t >= burn:
            acc += th.T @ th / n_chains; cnt += 1
    return acc / cnt


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    rng = np.random.default_rng(0)
    settings = [
        dict(kappa=10.0, gamma=1.0, phi=0.0, eta=0.02, B=8),
        dict(kappa=20.0, gamma=2.0, phi=np.deg2rad(30), eta=0.05, B=4),
        dict(kappa=5.0, gamma=0.0, phi=np.deg2rad(60), eta=0.1, B=1),
        dict(kappa=30.0, gamma=-0.5, phi=np.deg2rad(45), eta=0.01, B=2),
    ]
    worst = 0.0
    for s in settings:
        H, C = make_HC(s["kappa"], s["gamma"], s["phi"])
        eta, B = s["eta"], s["B"]
        etaB = eta / B
        Q = etaB * C
        S_c = lyap_cont_3x3(H, Q)
        S_c_sp = solve_continuous_lyapunov(H, Q)
        A = np.eye(2) - eta * H
        S_d = lyap_disc_3x3(A, eta ** 2 * C / B)
        S_d_sp = solve_discrete_lyapunov(A, eta ** 2 * C / B)
        print(f"\n== kappa={s['kappa']} gamma={s['gamma']} phi={np.rad2deg(s['phi']):.0f}deg eta={eta} B={B}")
        print(f"  3x3 cont vs scipy rel err {rel(S_c, S_c_sp):.1e};  3x3 disc vs scipy rel err {rel(S_d, S_d_sp):.1e}")
        worst = max(worst, rel(S_c, S_c_sp), rel(S_d, S_d_sp))
        if s["phi"] == 0.0:
            S_cf = etaB / 2 * C @ np.linalg.inv(H)
            print(f"  commuting closed form (eta/2B) C H^-1 vs Lyapunov rel err {rel(S_cf, S_c):.1e}")
            worst = max(worst, rel(S_cf, S_c))
        er = 0.5 * np.trace(H @ S_c)
        er_formula = etaB * np.trace(C) / 4
        print(f"  excess risk 1/2Tr(HS) = {er:.6e}, eta Tr C/(4B) = {er_formula:.6e}")
        worst = max(worst, abs(er - er_formula) / er_formula)
        # Euler-Maruyama of the SDE (dt small relative to 1/kappa)
        dt = 0.02 / s["kappa"]
        n_steps = int(40 / dt)  # 40 time units ~ 40 relaxation times of slow mode
        S_em = em_sde(H, C, etaB, dt, n_steps, 400, rng)
        # EM itself has O(dt) bias; compare against discrete Lyapunov with step dt too
        S_em_exact = lyap_disc_3x3(np.eye(2) - dt * H, etaB * dt * C)
        print(f"  SDE Euler-Maruyama MC vs continuous Lyapunov: rel err {rel(S_em, S_c):.3f} "
              f"(EM's own dt-bias would be {rel(S_em_exact, S_c):.3f})")
        # direct discrete SGD
        n_steps = int(60 / eta)
        S_mc = sim_sgd(H, C, eta, B, n_steps, 400, rng)
        print(f"  discrete SGD MC vs discrete Lyapunov: rel err {rel(S_mc, S_d):.3f};  "
              f"SDE Lyapunov vs discrete Lyapunov: rel diff {rel(S_c, S_d):.3f}")
        print(f"  discrete excess risk {0.5*np.trace(H@S_d):.4e}  (SDE {er:.4e}; MC {0.5*np.trace(H@S_mc):.4e})")
        # KL*
        k, Tstar = kl_star(H, S_c)
        res = minimize_scalar(lambda lt: kl_gauss(S_c, np.exp(lt) / N_DATA * np.linalg.inv(H)),
                              bounds=(-20, 20), method="bounded", options=dict(xatol=1e-10))
        print(f"  KL* closed form {k:.6f} at T*={Tstar:.5f};  numeric min {res.fun:.6f} at T={np.exp(res.x):.5f}")
        print(f"  check: posterior excess risk at T*  T*p/(2n) = {Tstar*2/(2*N_DATA):.6e} == SGD excess risk {er:.6e}")
        worst = max(worst, abs(k - res.fun))

    # Shape intuition: commuting case, Sigma proportional to H^(gamma-1)
    print("\n== shape check, kappa=10, phi=0: eigen-ratio Sigma_22/Sigma_11 (should be kappa^(gamma-1))")
    for g in [-1, 0, 1, 2, 3]:
        H, C = make_HC(10.0, g, 0.0)
        S = lyap_cont_3x3(H, C)
        k, _ = kl_star(H, S)
        print(f"  gamma={g:+d}: ratio {S[1,1]/S[0,0]:.4g}  kappa^(g-1)={10.0**(g-1):.4g}  KL*={k:.4f}")
    print(f"\nworst algebraic discrepancy: {worst:.2e}")

    # reference numbers for the JS self-test
    H, C = make_HC(20.0, 2.0, np.deg2rad(30))
    S_c = lyap_cont_3x3(H, 0.05 / 4 * C)
    S_d = lyap_disc_3x3(np.eye(2) - 0.05 * H, 0.05 ** 2 * C / 4)
    print("\nJS reference (kappa=20,gamma=2,phi=30,eta=.05,B=4):")
    print("  S_c =", S_c.ravel().tolist())
    print("  S_d =", S_d.ravel().tolist())
    print("  KL* =", kl_star(H, S_c))


if __name__ == "__main__":
    main()
