"""Checks for superposition.html (Elhage et al. 2022 toy model, n=2 features, m=1 dim).

x_i = 0 w.p. S, else U[0,1] (density p = 1-S), x' = ReLU(W^T W x + b), loss = sum_i I_i (x_i - x'_i)^2,
I_1 = 1, I_2 = r.  Candidates:
  A  "feature 2 ignored"   W=[1,0], b=[0, p/2]:    L_A = r * Var(x2) = r (p/3 - p^2/4)
  B  "feature 1 ignored"   W=[0,1], b=[p/2, 0]:    L_B =     (p/3 - p^2/4)
  C  "antipodal"           W=[a,-c], a,c>0, both biases optimised: L_C(p,r) = local minimum found by
                           Nelder-Mead continued from the sparse side (it disappears, c->0, in the dense phase).
     The special case a=c (w2=-w1) gives L = (1+r) G(p); a=c=1, b=0 gives (1+r) p^2/6 exactly.
  Importances (1,r) and (r,1) are related by swapping the features, so only r<=1 is computed.
The per-feature expected loss for x'_i = ReLU(alpha x_i - beta x_j + b):
  q^2 ReLU(b)^2 + p q E1(alpha,b) + p q E2(beta,b) + p^2 E3(alpha,beta,b),
with the inner v-integral of E3 done in closed form and the outer u-integral by piecewise
Gauss-Legendre (exact, the pieces are polynomials).
"""
import numpy as np
from scipy.optimize import minimize

GL_X, GL_W = np.polynomial.legendre.leggauss(12)

def _pieces(bps):
    pts = sorted(set([0.0, 1.0] + [t for t in bps if 0.0 < t < 1.0]))
    return list(zip(pts[:-1], pts[1:]))

def _gl(f, bps):
    s = 0.0
    for lo, hi in _pieces(bps):
        u = 0.5 * (hi - lo) * GL_X + 0.5 * (hi + lo)
        s += 0.5 * (hi - lo) * np.sum(GL_W * f(u))
    return s

def E1(al, b):   # E_u (u - ReLU(al u + b))^2
    return _gl(lambda u: (u - np.maximum(al * u + b, 0)) ** 2, [-b / al if al else 2])

def E2(be, b):   # E_v ReLU(b - be v)^2
    if b <= 0: return 0.0
    t = 1.0 if be <= 0 else min(1.0, b / be)
    return b * b * t - b * be * t * t + be * be * t ** 3 / 3

def E3(al, be, b):  # E_{u,v} (u - ReLU(al u - be v + b))^2, be >= 0
    def f(u):
        c = al * u + b
        t = np.where(c <= 0, 0.0, np.minimum(1.0, c / be) if be > 0 else 1.0)
        d = u - c
        return d * d * t + d * be * t * t + be * be * t ** 3 / 3 + (1 - t) * u * u
    bps = [-b / al, (be - b) / al] if al else []
    return _gl(f, bps)

def feat_loss(p, al, be, b):
    q = 1 - p
    return q * q * max(b, 0) ** 2 + p * q * (E1(al, b) + E2(be, b)) + p * p * E3(al, be, b)

def V(p): return p / 3 - p * p / 4

def anti_loss(th, p, r):
    la, lc, b1, b2 = th
    a, c = np.exp(la), np.exp(lc)
    return feat_loss(p, a * a, a * c, b1) + r * feat_loss(p, c * c, a * c, b2)

def G(p):
    """Per-feature loss of the symmetric antipodal solution W=[a,-a], optimal a and biases.
    Both features see the same loss function, so L_C(p,r) = (1+r) G(p) exactly."""
    from scipy.optimize import minimize_scalar
    def inner(la):
        a2 = np.exp(2 * la)
        return minimize_scalar(lambda b: feat_loss(p, a2, a2, b), bounds=(-0.5, 0.5),
                               method="bounded", options=dict(xatol=1e-10)).fun
    res = minimize_scalar(inner, bounds=(-0.5, 0.5), method="bounded", options=dict(xatol=1e-8))
    a2 = np.exp(2 * res.x)
    bb = minimize_scalar(lambda b: feat_loss(p, a2, a2, b), bounds=(-0.5, 0.5), method="bounded",
                         options=dict(xatol=1e-10)).x
    return res.fun, np.sqrt(a2), bb

NM = dict(xatol=1e-9, fatol=1e-15, maxiter=20000)

def antipodal_row(r, ps):
    """Continuation in p (sparse -> dense) of the antipodal local minimum for importance r<=1.
    Returns list of (L_C, a, c, b1, b2), with L_C = nan once the minimum has collapsed (c/a < 0.03)."""
    th = np.array([0., 0., 0., 0.]); out = []; alive = True
    for p in ps:
        if alive:
            res = minimize(anti_loss, th, args=(p, r), method="Nelder-Mead", options=NM)
            th = res.x; a, c = np.exp(th[:2])
            if c / a < 0.03: alive = False
        out.append((res.fun if alive else np.nan, *(np.r_[np.exp(th[:2]), th[2:]] if alive else [np.nan] * 4)))
    return out

def phase(p, r, LC):
    """'A' feature 2 ignored, 'B' feature 1 ignored, 'C' antipodal superposition."""
    L = dict(A=r * V(p), B=V(p))
    if not np.isnan(LC): L["C"] = LC
    return min(L, key=L.get), L

def p_star_b0(r):
    """Boundary with b=0, a=1: (1+r) p^2/6 = min(r,1) (p/3 - p^2/4)."""
    k = min(r, 1.0)
    return (k / 3) / ((1 + r) / 6 + k / 4)

def mc_loss(W, b, p, I, N=4_000_000, seed=0):
    """Direct Monte Carlo of the expected loss, in chunks."""
    rng = np.random.default_rng(seed); tot = 0.0; done = 0
    while done < N:
        k = min(2_000_000, N - done)
        x = rng.random((k, W.shape[1])) * (rng.random((k, W.shape[1])) < p)
        y = np.maximum(x @ W.T @ W + b, 0)
        tot += float(((x - y) ** 2 @ I).sum()); done += k
    return tot / N

P_GRID = np.logspace(-2, 0, 81)          # density 1-S
R_HALF = np.logspace(-1, 0, 41)          # relative importance r <= 1 (r > 1 by symmetry)

def _row(r): return antipodal_row(r, P_GRID)

if __name__ == "__main__":
    import json
    from multiprocessing import Pool
    np.set_printoptions(precision=5, suppress=True)
    out = {}
    print("== (a) closed-form expected loss vs Monte Carlo (2e7 samples) ==")
    for (p, r, th) in [(0.1, 1.0, [0.0, 0.0, -0.1, -0.1]), (0.3, 0.5, [0.1, -0.2, 0.05, -0.2]),
                       (0.05, 2.0, [-0.1, 0.05, -0.02, 0.03])]:
        a, c = np.exp(th[0]), np.exp(th[1])
        W = np.array([[a, -c]]); b = np.array(th[2:]); I = np.array([1, r])
        f, mc = anti_loss(th, p, r), mc_loss(W, b, p, I, N=20_000_000, seed=11)
        print(f"  antipodal p={p} r={r} W={W.ravel()} b={b}: formula {f:.6f}  MC {mc:.6f}  rel.diff {mc/f-1:+.2%}")
    for p, r in [(0.2, 0.7), (0.6, 1.5)]:
        f, mc = r * V(p), mc_loss(np.array([[1., 0.]]), np.array([0, p / 2]), p, np.array([1, r]), N=20_000_000)
        print(f"  ignore-f2  p={p} r={r}: formula {f:.6f}  MC {mc:.6f}  rel.diff {mc/f-1:+.2%}")

    print("\n== (b) phase table: antipodal local minimum by continuation, r in [0.1,1] x p in [0.01,1] ==")
    with Pool(12) as pool: rows = pool.map(_row, list(R_HALF))
    LC = np.array([[c[0] for c in row] for row in rows])          # [r, p]
    AC = np.array([[c[1:] for c in row] for row in rows])
    bnd = []
    for i, r in enumerate(R_HALF):
        ph = [phase(p, r, LC[i, j])[0] for j, p in enumerate(P_GRID)]
        j = next((j for j, q in enumerate(ph) if q != "C"), len(ph))
        bnd.append(P_GRID[j - 1] if j > 0 else np.nan)
    for i in [0, 10, 20, 30, 40]:
        r = R_HALF[i]
        print(f"  r={r:.3f}: superposition for p <= {bnd[i]:.4f} (grid);  symmetric a=c,b=0 closed form p*={p_star_b0(r):.4f}")
    for i, j in [(0, 40), (20, 60), (40, 70)]:
        print(f"  sample cell r={R_HALF[i]:.3f} p={P_GRID[j]:.4f}: L_C={LC[i,j]:.6e} a,c,b1,b2={AC[i,j]}")
    out["p_grid"] = P_GRID.tolist(); out["r_half"] = R_HALF.tolist()
    out["LC"] = [[None if np.isnan(v) else float(f"{v:.6e}") for v in row] for row in LC]
    out["AC"] = [[None if np.isnan(v[0]) else [round(float(x), 5) for x in v] for v in row] for row in AC]

    print("\n== (c) symmetric w2=-w1 solution: G(p) vs p^2/6 ==")
    for p in [0.01, 0.1, 0.3, 1.0]:
        g, a, b = G(p); print(f"  p={p}: G={g:.6e} (p^2/6={p*p/6:.6e}) a={a:.4f} b={b:+.5f}")

    print("\n== (d) empirical phase diagram: train 2->1 models with Adam (torch), 6 inits per cell ==")
    import torch
    torch.manual_seed(0)
    pj = list(range(0, 81, 5)); ri = list(range(0, 81, 10))   # r index into full grid of 81 (0.1..10)
    R_FULL = np.logspace(-1, 1, 81); K = 6
    cellsPR = [(P_GRID[j], R_FULL[i], j, i) for j in pj for i in ri]
    P = np.repeat([c[0] for c in cellsPR], K); R = np.repeat([c[1] for c in cellsPR], K); M = len(P)
    Pt = torch.tensor(P, dtype=torch.float32)[:, None, None]
    It = torch.stack([torch.ones(M), torch.tensor(R, dtype=torch.float32)], 1)[:, None, :]
    W = (torch.randn(M, 1, 2) * np.sqrt(2 / 3)).requires_grad_(); b = torch.zeros(M, 1, 2, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2); steps, B = 5000, 2048
    for t in range(steps):
        for gq in opt.param_groups: gq["lr"] = 1e-2 * 0.5 * (1 + np.cos(np.pi * t / steps)) + 1e-4
        x = torch.rand(M, B, 2) * (torch.rand(M, B, 2) < Pt)
        y = torch.relu(x @ W.transpose(1, 2) @ W + b)
        loss = ((((x - y) ** 2) * It).sum(-1).mean(1) / Pt.ravel() / torch.maximum(It[:, 0, 1], torch.ones(M))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    Wn, bn = W.detach().numpy()[:, 0], b.detach().numpy()[:, 0]
    cells = []; agree = 0
    def theory(p, r, j, i):
        if r <= 1: lc = LC[i, j]
        else: lc = r * LC[80 - i, j]   # swap features: importances (1,r) = r * (1/r, 1)
        return phase(p, r, lc)
    for ci, (p, r, j, i) in enumerate(cellsPR):
        Ls = [mc_loss(Wn[ci*K+k][None], bn[ci*K+k], p, np.array([1, r]), N=400_000, seed=3) for k in range(K)]
        k = int(np.argmin(Ls)); w = Wn[ci * K + k]
        got = "C" if w[0] * w[1] < 0 and min(abs(w)) > 0.1 * max(abs(w)) else ("A" if abs(w[0]) > abs(w[1]) else "B")
        pred, L = theory(p, r, j, i)
        agree += got == pred
        cells.append(dict(p=round(float(p), 5), r=round(float(r), 5), got=got, pred=pred, w=[round(float(v), 3) for v in w]))
        if got != pred:
            print(f"  mismatch p={p:.3f} r={r:.3f}: trained {got} W={w} L_MC={Ls[k]:.4e}; theory {pred} " +
                  " ".join(f"L_{q}={v:.4e}" for q, v in L.items()))
    print(f"  trained phase == theoretical phase in {agree}/{len(cells)} cells")
    out["trained"] = cells
    json.dump(out, open("superposition_check_out.json", "w"))

    print("\n== (e) panel 2: n=5, m=2, I_i = 0.9^i, Adam lr 1e-2 cosine, batch 1024, 3000 steps ==")
    n, m = 5, 2
    for S in [0.0, 0.8, 0.95, 0.99]:
        p = 1 - S; I = 0.9 ** torch.arange(n).float(); sd = 4
        W = (torch.randn(sd, m, n) * np.sqrt(2 / (m + n))).requires_grad_(); b = torch.zeros(sd, n, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2)
        for t in range(3000):
            for gq in opt.param_groups: gq["lr"] = 1e-2 * 0.5 * (1 + np.cos(np.pi * t / 3000))
            x = torch.rand(sd, 1024, n) * (torch.rand(sd, 1024, n) < p)
            y = torch.relu(torch.einsum("smn,smk,sbk->sbn", W, W, x) + b[:, None])
            loss = (((x - y) ** 2) * I).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        Wd = W.detach().numpy()
        for s in range(sd):
            nr = np.linalg.norm(Wd[s], axis=0); on = nr > 0.5
            ang = np.sort(np.degrees(np.arctan2(Wd[s][1], Wd[s][0]))[on])
            gaps = np.diff(np.r_[ang, ang[0] + 360]) if on.sum() > 1 else []
            lm = mc_loss(Wd[s], b.detach().numpy()[s], p, (0.9 ** np.arange(n)), N=1_000_000, seed=2)
            print(f"  S={S} seed{s}: loss {lm:.5f}, represented {on.sum()} features, norms {np.round(nr,2)}, angular gaps {np.round(gaps)}")
