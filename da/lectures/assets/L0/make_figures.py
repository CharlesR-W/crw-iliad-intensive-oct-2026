"""Figures for the DA derivation deck, L0_deriving_the_notes.html (Louis Jaburi's notes, derived).

Run from anywhere:  python da/lectures/assets/L0/make_figures.py
CPU only (numpy, scipy, matplotlib); a few seconds. Writes PNGs next to this file and prints
the numbers the slides quote. The other figures in the deck are shared with L1 (assets/L1/).

Figures
  unroll_to_if.png  Exercise 4.2(b): full-batch GD on least squares; the unrolled response to
                    downweighting z_k, in the eigenbasis of H, converges to H^{-1} g_k (the IF).
  dog_cat.png       Exercise 4.3(c): exact linear response of a two-layer linear network's
                    gradient flow; influence of upweighting the dog on the cat's loss over time.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- style (as in L1, L2)
PAPER, INK, INK2, RULE = "#fcfbf7", "#1a1f26", "#4d5763", "#ddd7c9"
BLUE, ORANGE, GREEN, PURPLE, RED = "#1b6aa8", "#c0701a", "#0a8a5f", "#8a4fb0", "#b8442a"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 22, "axes.titlesize": 24, "axes.labelsize": 25,
    "legend.fontsize": 21, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# ---------------------------------------------------------------- unroll_to_if.png
def unroll_to_if():
    rng = np.random.default_rng(0)
    n = 20
    X = rng.normal(size=(n, 2)) * np.array([2.0, 0.35])
    y = X @ np.array([1.0, -2.0]) + 0.5 * rng.normal(size=n)
    H = X.T @ X                                   # summed-loss Hessian
    w_star = np.linalg.solve(H, X.T @ y)
    k = int(np.argmax(np.abs(X[:, 1])))           # a point that pushes along the flat direction
    grad_k = lambda w: -X[k] * (y[k] - X[k] @ w)
    alpha = 1.0 / np.linalg.eigvalsh(H).max()     # step on the summed loss
    T = 600
    w = np.array([-3.0, 3.0])
    r = np.zeros(2)
    rs = [r.copy()]
    for _ in range(T):
        r = (np.eye(2) - alpha * H) @ r + alpha * grad_k(w)   # Ex. 4.2(a), full batch
        w = w - alpha * (H @ w - X.T @ y)
        rs.append(r.copy())
    rs = np.array(rs)
    r_if = np.linalg.solve(H, grad_k(w_star))
    # finite-difference check of the response: GD with z_k downweighted by eps
    eps = 1e-6
    wp = np.array([-3.0, 3.0])
    for _ in range(T):
        g = H @ wp - X.T @ y - eps * grad_k(wp)
        wp = wp - alpha * g
    fd = (wp - w) / eps
    print(f"unroll_to_if: |r_T - H^-1 g_k| = {np.abs(rs[-1] - r_if).max():.2e}, "
          f"|finite diff - r_T| = {np.abs(fd - rs[-1]).max():.2e}")
    evals, evecs = np.linalg.eigh(H)
    proj = rs @ evecs
    proj_if = evecs.T @ r_if
    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = [f"flat direction ($\\sigma$ = {evals[0]:.1f})", f"stiff direction ($\\sigma$ = {evals[1]:.0f})"]
    for j, c in zip([1, 0], [BLUE, ORANGE]):
        ax.plot(proj[:, j], color=c, lw=2.6, label=labels[j])
        ax.axhline(proj_if[j], color=c, ls="--", lw=1.6)
    ax.set_xlabel("GD step $t$")
    ax.set_ylabel("component of $r_t$")
    ax.text(T * 0.55, 0.03, "dashed: $H^{-1}g_k$ (the IF limit)", color=INK2, fontsize=20)
    ax.legend(loc="lower right")
    save(fig, "unroll_to_if.png")


# ---------------------------------------------------------------- dog_cat.png
S1, S2, A0, TAU = 3.0, 1.0, 1e-8, 1.0
XS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)   # dog, cat, oak, rose
YS = XS @ np.diag([S1, S2])                                     # realizable, y = S x
N = len(XS)


def flow(beta, t_eval):
    Sxx = (XS.T * beta) @ XS / N
    Syx = (YS.T * beta) @ XS / N

    def rhs(t, v):
        W1, W2 = v[:4].reshape(2, 2), v[4:].reshape(2, 2)
        E = Syx - W2 @ W1 @ Sxx
        return np.concatenate([(W2.T @ E).ravel(), (E @ W1.T).ravel()]) / TAU

    v0 = np.concatenate([np.sqrt(A0) * np.eye(2).ravel()] * 2)
    sol = solve_ivp(rhs, (0, t_eval[-1]), v0, t_eval=t_eval, rtol=1e-11, atol=1e-14, method="DOP853")
    return [sol.y[4:, i].reshape(2, 2) @ sol.y[:4, i].reshape(2, 2) for i in range(len(t_eval))]


def dog_cat():
    t = np.linspace(0, 14, 1401)
    eps = 1e-5
    bp, bm = np.ones(N), np.ones(N)
    bp[0] += eps
    bm[0] -= eps
    loss_cat = lambda W: 0.5 * np.sum((YS[1] - W @ XS[1]) ** 2)
    Wp, Wm = flow(bp, t), flow(bm, t)
    infl = np.array([(loss_cat(a) - loss_cat(b)) / (2 * eps) for a, b in zip(Wp, Wm)])
    t1 = TAU / (2 * S1) * np.log(S1 / A0)
    t2 = TAU / (2 * S2) * np.log(S2 / A0)
    plateau = S1**2 * S2**2 / (N * (S1**2 - S2**2))
    mid = np.argmin(np.abs(t - 0.5 * (t1 + t2)))
    print(f"dog_cat: t1* = {t1:.2f}, t2* = {t2:.2f}; plateau prediction {plateau:.4f}, "
          f"simulated at midpoint {infl[mid]:.4f}; min {infl.min():.3f}, max {infl.max():.3f}, "
          f"final {infl[-1]:.1e}")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axhline(0, color=INK2, lw=1)
    ax.plot(t, infl, color=PURPLE, lw=2.8)
    ax.axhline(plateau, color=RED, ls="--", lw=1.6)
    lo = -0.9
    ax.set_ylim(lo, 0.75)
    ax.text(0.5 * (t1 + t2) - 1.2, plateau + 0.12, "plateau formula", color=RED, fontsize=20)
    imin = int(np.argmin(infl))
    ax.annotate(f"dips to {infl.min():.1f}", xy=(t[imin], lo), xytext=(t[imin] + 0.6, lo + 0.25),
                color=INK2, fontsize=19, arrowprops=dict(arrowstyle="->", color=INK2))
    for tt, lab in [(t1, "$t_1^*$"), (t2, "$t_2^*$")]:
        ax.axvline(tt, color=INK2, ls=":", lw=1.4)
        ax.text(tt + 0.1, 0.55, lab, color=INK2, fontsize=22)
    ax.set_xlabel("training time $t/\\tau$")
    ax.set_ylabel("$\\partial\\,\\ell_{\\mathrm{cat}}/\\partial\\beta_{\\mathrm{dog}}$")
    ax.text(10.8, -0.5, "above 0: hurts\nbelow 0: helps", color=INK2, fontsize=19)
    save(fig, "dog_cat.png")
    return dict(t1=t1, t2=t2, plateau=plateau, mid=infl[mid], final=infl[-1], mn=infl.min())


if __name__ == "__main__":
    unroll_to_if()
    dog_cat()
