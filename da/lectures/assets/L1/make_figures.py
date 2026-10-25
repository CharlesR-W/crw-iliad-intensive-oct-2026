"""Figures for DA lecture L1, "What did this training example do?".

Run from anywhere:  python da/lectures/assets/L1/make_figures.py
CPU only (numpy, scipy, matplotlib, torch); about 30 s. Writes PNGs next to this file
and prints the numbers the slides quote.

Figures
  cube.png        the three targets in data-weight space for n = 3: the derivative at 1,
                  leave-one-out at 1 - e_i, subset deletion at a far corner.
  response.png    the exact response of a regression slope to one example's weight beta_j,
                  for a high-leverage point: the tangent at beta = 1 is the influence function,
                  the value at beta = 0 is leave-one-out (DA1 Sec. 3 data).
  pull.png        the three factors of -g_phi^T H^{-1} g_i on an anisotropic quadratic bowl.
  wells.png       schematic: a reweighting tilts a double well and the global argmin jumps.
  leverage.png    DA1 Sec. 3: removing a low- and a high-leverage point, and exact / IF against h.
  valley.png      the a.b model: the hyperbola of minimizers, v across it and v_perp along it.
  ab_damping.png  DA1 Sec. 4: damped IF against lambda for four minimizers, phi = a^2 and phi = ab.
  bregman.png     the Bregman divergence of a convex output loss.
  proximity.png   DA1 Sec. 4 extension: damped IF against proximal and plain retraining, tiny MLP.
  filters.png     spectral filters: 1/s, 1/(s+lambda) and K steps of GD.
"""
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

T0 = time.time()
OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- style (as in L2)
PAPER, INK, INK2, RULE = "#fcfbf7", "#1a1f26", "#4d5763", "#ddd7c9"
PANEL = "#f3f0e8"
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


def arrow(ax, x0, y0, x1, y1, color, lw=2.6, ms=24):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0))


# ================================================================ cube.png
# Data-weight space for n = 3, drawn in an oblique projection.
def proj(p):
    x, y, z = p
    return np.array([x + 0.42 * y, z + 0.30 * y])


fig, ax = plt.subplots(figsize=(8.6, 6.4))
ax.set_aspect("equal"); ax.axis("off")
corners = [np.array(c) for c in np.ndindex(2, 2, 2)]
for a in corners:
    for b in corners:
        if np.sum(np.abs(a - b)) == 1 and tuple(a) < tuple(b):
            pa, pb = proj(a), proj(b)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=RULE, lw=2, zorder=1)
one, loo, sub = np.array([1, 1, 1]), np.array([0, 1, 1]), np.array([0, 0, 1])
p1, pl, ps = proj(one), proj(loo), proj(sub)
ax.plot(*p1, "o", color=INK, ms=16, zorder=5)
ax.text(p1[0] + 0.07, p1[1], r"$\mathbf{1}$: the real run", fontsize=28, color=INK, va="center")
arrow(ax, p1[0], p1[1], p1[0] - 0.33, p1[1], BLUE, lw=4)
ax.text(p1[0] - 0.17, p1[1] + 0.05, "derivative", fontsize=26, color=BLUE, ha="center", va="bottom")
ax.plot(*pl, "o", color=ORANGE, ms=15, zorder=5)
ax.text(pl[0] - 0.06, pl[1] + 0.07, r"leave-one-out $\mathbf{1}-e_1$", fontsize=26, color=ORANGE, ha="right")
ax.plot(*ps, "o", color=PURPLE, ms=15, zorder=5)
ax.text(ps[0] - 0.06, ps[1] - 0.02, "subset deletion:\n" + r"keep only $S=\{3\}$", fontsize=26,
        color=PURPLE, ha="right", va="top")
ax.text(proj([0, 0, 0])[0] + 0.03, proj([0, 0, 0])[1] - 0.09, r"$\beta=0$", fontsize=22, color=INK2)
for k, p in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
    q = proj(np.array(p) * 1.18) + (np.array([0.08, -0.05]) if k == 2 else 0)
    ax.text(q[0], q[1], rf"$\beta_{k + 1}$", fontsize=25, color=INK2, ha="center", va="center")
ax.set_xlim(-0.75, 2.45); ax.set_ylim(-0.25, 1.48)
save(fig, "cube.png")

# ================================================================ the DA1 Sec. 3 regression
rng = np.random.default_rng(0)
n = 15
x = np.r_[rng.normal(0, 1, n - 1), 7.0]
y = 1.0 + 0.6 * x + 0.4 * rng.normal(size=n)
y[-1] -= 3.0
Xd = np.c_[np.ones(n), x]
A = Xd.T @ Xd
w = np.linalg.solve(A, Xd.T @ y)
r = y - Xd @ w
h = np.einsum("ij,jk,ik->i", Xd, np.linalg.inv(A), Xd)
pred = -(np.linalg.solve(A, Xd.T) * r).T
exact = np.array([np.linalg.lstsq(np.delete(Xd, j, 0), np.delete(y, j), rcond=None)[0] - w
                  for j in range(n)])
jA, jB = int(np.argmin(np.abs(x[:-1]))), n - 1
print(f"regression: h_A = {h[jA]:.3f}, h_B = {h[jB]:.3f}, 1/(1-h_B) = {1 / (1 - h[jB]):.2f}, "
      f"sum h = {h.sum():.3f}; exact = pred/(1-h) to {np.abs(exact - pred / (1 - h)[:, None]).max():.1e}")

# --- response.png: slope of the fit as a function of beta_B (closed form, DA1 Ex. 3.2)
bb = np.linspace(0, 1, 300)
qB = np.linalg.solve(A, Xd[jB])
resp = np.array([w[1] - (1 - b) * qB[1] * r[jB] / (1 - (1 - b) * h[jB]) for b in bb])
slope_IF = qB[1] * r[jB]                       # d slope / d beta_B = (A^{-1} x_B r_B)_slope
for b in [0.0, 0.5]:                             # brute-force check of the closed form
    wt = np.ones(n); wt[jB] = b
    wb = np.linalg.solve((Xd.T * wt) @ Xd, Xd.T @ (wt * y))
    assert abs(wb[1] - resp[np.argmin(np.abs(bb - b))]) < 2e-3
fig, ax = plt.subplots(figsize=(9.6, 5.4))
ax.plot(bb, resp, color=INK, lw=3, label="refit with weight $\\beta_B$")
tang = w[1] + slope_IF * (bb - 1)
ax.plot(bb, tang, color=BLUE, lw=2.6, ls="--", label="tangent at $\\beta_B=1$ (the IF)")
ax.plot(1, w[1], "o", color=INK, ms=12, zorder=5)
ax.plot(0, resp[0], "o", color=ORANGE, ms=13, zorder=5)
ax.plot(0, tang[0], "o", color=BLUE, ms=13, zorder=5)
ax.annotate("leave-one-out", xy=(0, resp[0]), xytext=(0.10, resp[0] + 0.02), fontsize=25, color=ORANGE,
            va="center")
ax.annotate("IF prediction", xy=(0, tang[0]), xytext=(0.06, tang[0] - 0.045), fontsize=25, color=BLUE,
            va="center")
ax.set_xlabel(r"weight $\beta_B$ of the far point B")
ax.set_ylabel("slope of the fitted line")
ax.legend(loc="upper right")
print(f"response: slope {w[1]:.3f} at beta=1; LOO {resp[0]:.3f}; IF prediction {tang[0]:.3f}; "
      f"ratio of changes {(resp[0] - w[1]) / (tang[0] - w[1]):.2f}")
save(fig, "response.png")

# --- leverage.png
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 2)
ax[0].scatter(x, y, s=40, c=INK, zorder=3)
ax[0].plot(xs, w[0] + w[1] * xs, color=INK, lw=3, zorder=4, label="fit on all points")
for j, name, col in [(jA, "A", BLUE), (jB, "B", RED)]:
    ax[0].annotate(name, (x[j], y[j]), xytext=(8, -18), textcoords="offset points", color=col,
                   weight="bold", fontsize=28)
    we, wp = w + exact[j], w + pred[j]
    ax[0].plot(xs, we[0] + we[1] * xs, "-", color=col, lw=2, label="point removed: refit" if name == "A" else None)
    ax[0].plot(xs, wp[0] + wp[1] * xs, ":", color=col, lw=3, label="point removed: IF" if name == "A" else None)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
ax[0].legend(fontsize=20, loc="upper left")
ax[0].text(7.4, w[0] + exact[jB][0] + (w[1] + exact[jB][1]) * 7.4 + 0.2, "B removed", color=RED, fontsize=20, ha="right", va="bottom")
ratio = exact[:, 1] / pred[:, 1]
hh = np.linspace(0, 0.9, 200)
ax[1].plot(hh, 1 / (1 - hh), "--", color=INK2, lw=1.6, label="$1/(1-h)$")
ax[1].scatter(h, ratio, s=60, color=INK, zorder=3)
for j, name, col in [(jA, "A", BLUE), (jB, "B", RED)]:
    ax[1].annotate(name, (h[j], ratio[j]), xytext=(8, -20), textcoords="offset points", color=col,
                   weight="bold", fontsize=28)
ax[1].set_xlabel("leverage $h_j$"); ax[1].set_ylabel("exact / IF change in slope")
ax[1].legend(loc="upper left")
plt.tight_layout()
save(fig, "leverage.png")

# ================================================================ pull.png
ang = np.deg2rad(30)
U = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
evs = np.array([1.2, 0.25])                     # stiff along U[:,0], flat along U[:,1]
H2 = U @ np.diag(evs) @ U.T
pull = 1.0 * U[:, 0] + 0.3 * U[:, 1]          # the pull -g_i: mostly along the stiff axis
step = np.linalg.solve(H2, pull)               # the move -H^{-1} g_i
fig, ax = plt.subplots(figsize=(9.6, 6.4))
ax.set_aspect("equal"); ax.axis("off")
t = np.linspace(0, 2 * np.pi, 300)
for c in [0.05, 0.2, 0.45, 0.8]:
    e = U @ np.vstack([np.sqrt(2 * c / evs[0]) * np.cos(t), np.sqrt(2 * c / evs[1]) * np.sin(t)])
    ax.plot(e[0], e[1], color=RULE, lw=1.8)
fl = 2.6 * U[:, 1]
ax.plot([-fl[0], fl[0]], [-fl[1], fl[1]], ls=":", color=INK2, lw=1.5)
ax.text(-fl[0] + 0.08, -fl[1] + 0.05, "flat direction", fontsize=24, color=INK2, ha="left", va="bottom")
st = 2.0 * U[:, 0]
ax.plot([-st[0], st[0]], [-st[1], st[1]], ls=":", color=INK2, lw=1.5)
ax.text(-st[0], -st[1] - 0.08, "stiff direction", fontsize=24, color=INK2, ha="left", va="top")
ax.plot(0, 0, "o", color=INK, ms=11, zorder=5)
ax.text(-0.12, -0.08, r"$w^*$", fontsize=28, ha="right", va="top")
arrow(ax, 0, 0, *pull, ORANGE, lw=3.4)
ax.text(pull[0] + 0.1, pull[1] - 0.05, r"pull: $-g_i$", fontsize=28, color=ORANGE, va="top")
arrow(ax, 0, 0, *step, BLUE, lw=3.4)
ax.text(step[0] - 0.1, step[1] + 0.05, r"move: $-H^{-1}g_i$", fontsize=28, color=BLUE, ha="right", va="bottom")
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.7, 2.3)
print(f"pull: angle between -g and -H^-1 g: "
      f"{np.degrees(np.arccos(pull @ step / np.linalg.norm(pull) / np.linalg.norm(step))):.0f} deg")
save(fig, "pull.png")

# ================================================================ wells.png (schematic)
ww = np.linspace(-1.65, 1.65, 400)
base = (ww ** 2 - 1) ** 2
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for k, (tilt, title) in enumerate([(-0.25, r"$\beta=\mathbf{1}$"), (0.25, r"$\beta_3$ lowered")]):
    Lw = base + tilt * ww
    ax[k].plot(ww, Lw, color=INK, lw=3)
    m = np.argmin(Lw)
    ax[k].plot(ww[m], Lw[m], "o", color=ORANGE, ms=14, zorder=5)
    ax[k].set_title(title, fontsize=28)
    ax[k].set_xlabel("$w$"); ax[k].set_yticklabels([]); ax[k].set_xticks([-1, 0, 1])
    ax[k].set_ylim(-0.45, 1.4)
loc = np.argmin(np.where(ww > 0, base + 0.25 * ww, np.inf))
ax[1].plot(ww[loc], (base + 0.25 * ww)[loc], "o", color=BLUE, ms=14, zorder=5)
ax[1].annotate("IF follows\nthis branch", xy=(ww[loc], (base + 0.25 * ww)[loc]), xytext=(0.15, 0.85),
               fontsize=22, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE))
ax[1].annotate("new argmin", xy=(ww[np.argmin(base + 0.25 * ww)], (base + 0.25 * ww).min()),
               xytext=(-1.55, 0.85), fontsize=22, color=ORANGE,
               arrowprops=dict(arrowstyle="-|>", color=ORANGE))
ax[0].set_ylabel("loss")
fig.text(0.5, -0.02, "tilt exaggerated", ha="center", fontsize=18, color=INK2)
plt.tight_layout()
save(fig, "wells.png")

# ================================================================ the a.b model
rng = np.random.default_rng(1)
z = 2.0 + rng.normal(size=10)
nz, zbar = len(z), z.mean()
i_ab = int(np.argmax(z))
print(f"a.b model: n = {nz}, zbar = {zbar:.3f}, z_i - zbar = {z[i_ab] - zbar:.3f}")

# --- valley.png
fig, ax = plt.subplots(figsize=(8.0, 6.4))
aa = np.linspace(0.45, 3.4, 300)
ax.plot(aa, zbar / aa, color=INK, lw=3)
ax.text(1.75, zbar / 1.75 + 0.6, r"$ab=\bar z$: all minimizers", fontsize=25, color=INK)
a0 = 1.0; b0 = zbar / a0
ax.plot(a0, b0, "o", color=INK, ms=12, zorder=5)
v = np.array([b0, a0]) / np.hypot(b0, a0)
vp = np.array([a0, -b0]) / np.hypot(a0, b0)
arrow(ax, a0, b0, a0 + 0.8 * v[0], b0 + 0.8 * v[1], BLUE, lw=3)
ax.text(a0 + 0.85 * v[0] + 0.05, b0 + 0.85 * v[1], r"$v=(b^*,a^*)$: curved, $g_i\parallel v$",
        fontsize=24, color=BLUE, va="bottom")
arrow(ax, a0, b0, a0 + 0.8 * vp[0], b0 + 0.8 * vp[1], ORANGE, lw=3)
ax.annotate(r"$v_\perp$: flat, $H v_\perp=0$", xy=(a0 + 0.8 * vp[0], b0 + 0.8 * vp[1] - 0.05), xytext=(1.45, 0.35),
            fontsize=24, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
ax.set_xlabel("$a$"); ax.set_ylabel("$b$")
ax.set_xlim(0, 3.6); ax.set_ylim(0, 3.6); ax.set_aspect("equal")
save(fig, "valley.png")

# --- ab_damping.png
lams = np.logspace(-3, 3, 200)


def warm_gf_endpoint(a0, b0, beta_i):
    wts = np.ones(nz); wts[i_ab] = beta_i

    def rhs(t, wv):
        gg = -np.sum(wts * (z - wv[0] * wv[1]))
        return [-gg * wv[1], -gg * wv[0]]
    return solve_ivp(rhs, (0, 5), [a0, b0], rtol=1e-11, atol=1e-13).y[:, -1]


fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
cols = [BLUE, GREEN, ORANGE, PURPLE]
for k, a0 in enumerate([0.6, 1.0, np.sqrt(zbar), 2.5]):
    b0 = zbar / a0
    vv = np.array([b0, a0])
    Hab = nz * np.outer(vv, vv)
    g_i = -(z[i_ab] - zbar) * vv
    for axk, gphi, phi in [(ax[0], np.array([2 * a0, 0.0]), lambda wv: wv[0] ** 2),
                           (ax[1], vv, lambda wv: wv[0] * wv[1])]:
        I_lam = [-gphi @ np.linalg.solve(Hab + lam * np.eye(2), g_i) for lam in lams]
        axk.semilogx(lams, I_lam, color=cols[k], lw=2.6, label=f"$(a^*,b^*)=({a0:.2f},{b0:.2f})$")
        eps = 1e-4
        fd = (phi(warm_gf_endpoint(a0, b0, 1 + eps)) - phi(warm_gf_endpoint(a0, b0, 1 - eps))) / (2 * eps)
        axk.plot(lams[0], fd, "o", mfc="none", mec=cols[k], mew=2, ms=13)
ax[0].set_title(r"$\phi=a^2$: a different limit for each minimizer")
ax[1].set_title(r"$\phi=ab$: one limit, $(z_i-\bar z)/n$")
for axk in ax:
    axk.set_xlabel(r"damping $\lambda$"); axk.set_ylabel(r"damped IF")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=19, columnspacing=1.2)
plt.tight_layout()
print(f"ab: phi=ab limit (z_i - zbar)/n = {(z[i_ab] - zbar) / nz:.4f}")
save(fig, "ab_damping.png")

# ================================================================ bregman.png
ff = np.linspace(-2.2, 2.6, 300)
ell = np.logaddexp(0, ff)                       # a convex output loss (softplus)
fs, f1 = -0.6, 1.8
sig = 1 / (1 + np.exp(-fs))
tangent = np.logaddexp(0, fs) + sig * (ff - fs)
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.plot(ff, ell, color=INK, lw=3, label=r"output loss $\ell(f)$")
ax.plot(ff, tangent, color=INK2, lw=1.8, ls="--", label=r"tangent at $f^s$")
ax.plot(fs, np.logaddexp(0, fs), "o", color=INK, ms=12, zorder=5)
ax.text(fs, np.logaddexp(0, fs) - 0.35, r"$f^s$", fontsize=28, ha="center")
y1, y0 = np.logaddexp(0, f1), np.logaddexp(0, fs) + sig * (f1 - fs)
ax.annotate("", xy=(f1, y1), xytext=(f1, y0),
            arrowprops=dict(arrowstyle="<|-|>", color=PURPLE, lw=2.6, mutation_scale=20, shrinkA=0, shrinkB=0))
ax.text(f1 + 0.1, (y0 + y1) / 2, r"$D(f,f^s)$", fontsize=28, color=PURPLE, va="center")
ax.plot(f1, y1, "o", color=PURPLE, ms=10)
ax.text(f1, y1 + 0.18, r"$f$", fontsize=28, ha="center", color=PURPLE)
ax.set_xlabel("network output $f$"); ax.set_yticklabels([]); ax.set_xticklabels([])
ax.legend(loc="upper left")
save(fig, "bregman.png")

# ================================================================ filters.png
s = np.logspace(-3, 2, 400)
lam_f, eta, K = 1.0, 0.01, 100                  # eta K = 1, crossover at s = 1
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.loglog(s, 1 / s, color=INK, lw=3, label=r"classical IF: $1/s$")
ax.loglog(s, 1 / (s + lam_f), color=BLUE, lw=2.8, label=r"damped: $1/(s+\lambda)$")
ax.loglog(s, (1 - (1 - eta * s) ** K) / s, color=ORANGE, lw=2.8, ls="--",
          label=r"$K$ GD steps: $[1-(1-\eta s)^K]/s$")
ax.set_xlabel(r"curvature eigenvalue $s$"); ax.set_ylabel(r"filter $F(s)$")
ax.set_ylim(1e-2, 1e3)
ax.legend(loc="upper right", fontsize=22)
ax.text(1.3e-3, 0.1, r"flat directions: $F(0)=1/\lambda$ or $\eta K$", fontsize=21, color=INK2)
save(fig, "filters.png")

# ================================================================ proximity.png (tiny MLP, as DA1 Sec. 4)
import torch
from torch.func import grad, hessian, jacrev, vmap

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
t0 = time.time()
n_mlp, width = 20, 8
Xt = torch.linspace(-2, 2, n_mlp)[:, None]
Yt = torch.sin(2 * Xt[:, 0]) + 0.2 * torch.randn(n_mlp)
xq, yq = torch.tensor([[0.5]]), torch.sin(torch.tensor([1.0]))
shapes = [(width, 1), (width,), (1, width), (1,)]
sizes = [int(np.prod(sh)) for sh in shapes]
P = sum(sizes)


def net(th, xx):
    W1, b1, W2, b2 = [q.reshape(sh) for q, sh in zip(torch.split(th, sizes), shapes)]
    return (torch.tanh(xx @ W1.T + b1) @ W2.T + b2).squeeze(-1)


ridge = 1e-3


def objective(th, beta, lam, center):
    return ((beta * 0.5 * (net(th, Xt) - Yt) ** 2).sum() + 0.5 * ridge * (th ** 2).sum()
            + 0.5 * lam * ((th - center) ** 2).sum())


v_obj = vmap(objective, in_dims=(0, 0, None, None))
v_grad = vmap(grad(objective), in_dims=(0, 0, None, None))
v_hess = vmap(hessian(objective), in_dims=(0, 0, None, None))


def newton(TH, B, lam, center, iters=100):
    for attempt in range(4):
        mu = torch.full((len(TH),), 1e-3)
        val = v_obj(TH, B, lam, center)
        for _ in range(iters):
            gg = v_grad(TH, B, lam, center)
            if gg.norm(dim=1).max() < 1e-7:
                break
            ev, V = torch.linalg.eigh(v_hess(TH, B, lam, center))
            d = torch.maximum(ev.abs(), mu[:, None])
            stp = (V @ ((V.transpose(1, 2) @ gg[..., None])[..., 0] / d)[..., None])[..., 0]
            new = TH - stp
            nv = v_obj(new, B, lam, center)
            ok = nv < val
            TH, val = torch.where(ok[:, None], new, TH), torch.where(ok, nv, val)
            mu = torch.where(ok, (mu / 3).clamp(min=1e-12), mu * 10)
        ev, V = torch.linalg.eigh(v_hess(TH, B, lam, center))
        saddle = ev[:, 0] < 0
        if not saddle.any():
            break
        kick = 0.05 * V[:, :, 0]
        better = v_obj(TH + kick, B, lam, center) < v_obj(TH - kick, B, lam, center)
        TH = torch.where(saddle[:, None], TH + torch.where(better[:, None], kick, -kick), TH)
    gg = v_grad(TH, B, lam, center)
    return TH, gg.norm(dim=1).max().item(), ev[:, 0].min().item()


phi_t = lambda th: 0.5 * ((net(th, xq) - yq) ** 2).sum()
zero = torch.zeros(P)
w_star, gnorm, _ = newton(0.8 * torch.randn(1, P), torch.ones(1, n_mlp), 0.0, zero, 300)
w_star = w_star[0]
Hm = hessian(objective)(w_star, torch.ones(n_mlp), 0.0, zero)
evm = torch.linalg.eigvalsh(Hm)
lam = 0.01 * evm[-1].item()
g_tr = jacrev(lambda t: 0.5 * (net(t, Xt) - Yt) ** 2)(w_star)
g_q = grad(phi_t)(w_star)
predm = (g_tr @ torch.linalg.solve(Hm + lam * torch.eye(P), g_q)).numpy()
B = 1 - torch.eye(n_mlp)
W_prox, gp, ep = newton(w_star.repeat(n_mlp, 1), B, lam, w_star)
W_plain, gu, eu = newton(w_star.repeat(n_mlp, 1), B, 0.0, w_star, 300)
prox = (vmap(phi_t)(W_prox) - phi_t(w_star)).numpy()
plain = (vmap(phi_t)(W_plain) - phi_t(w_star)).numpy()
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
rhos = []
for axk, target, name, col in [(ax[0], prox, "proximal retraining", BLUE), (ax[1], plain, "plain retraining", ORANGE)]:
    rho = spearmanr(predm, target)[0]
    rhos.append(rho)
    axk.scatter(predm, target, s=70, color=col, zorder=3)
    axk.axhline(0, color=INK2, lw=0.8); axk.axvline(0, color=INK2, lw=0.8)
    axk.set_xlabel("damped-IF prediction"); axk.set_ylabel(r"actual change in $\phi$")
    axk.set_title(f"{name}: Spearman {rho:.2f}")
plt.tight_layout()
print(f"tiny MLP: P = {P}, Hessian eigenvalues {evm[0]:.1e} ... {evm[-1]:.1f}, lambda = {lam:.3f}; "
      f"Spearman prox {rhos[0]:.2f}, plain {rhos[1]:.2f}; "
      f"size ratio prox {np.linalg.norm(prox) / np.linalg.norm(predm):.1f}, plain {np.linalg.norm(plain) / np.linalg.norm(predm):.1f}; "
      f"{time.time() - t0:.0f} s")
save(fig, "proximity.png")

print(f"total runtime {time.time() - T0:.1f}s")
