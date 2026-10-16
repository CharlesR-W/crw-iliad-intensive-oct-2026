"""Figures for TD lecture L2, "The pragmatic picture".

Run from anywhere:  python td/lectures/assets/L2/make_figures.py
CPU only (numpy, matplotlib, torch); about 40 s. Writes PNGs next to this file
and prints the numbers quoted in the slide captions.

Each figure is a slide-sized restatement of a code cell in the NB2 notebooks
(the cell is named in the comment above it); the notebooks remain the source.

Figures
  valley.png      the one picture of the lecture: GD at eta*S = 1.9 bouncing across a
                  narrow valley while drifting along its floor.
  igr.png         td2_1 §1.2: GD vs gradient flow vs the IGR flow on L = (ab-1)^2/2.
  eos.png         td2_2 §2.1: sharpness of a small MLP under full-batch GD at three step
                  sizes, each rising to its own 2/eta and hovering there.
  catapult.png    td2_2 §2.7: the exact catapult maps at eta*lambda_0 = 1.5, 3, 4.1.
  floor_scaling.png  td2_3 §3.3: SGD's loss floor in a quadratic basin, invariant under
                  (eta, B) -> (k eta, k B) until eta*h is not small.
  floor_cut.png   td2_3 §3.3: the drop at a learning-rate cut, against the exact expectation.
  balls.png       td2_4 §4.1: steepest descent under three norms (unit balls).
  signnorm.png    td2_4 §4.5: ||sign(G)||_2 against width, low-rank vs dense G.
  wd.png          td2_5 §5.2: the weight-decay equilibrium angle on a scale-invariant toy.
  spectrum.png    td2_6 §6.3: Hessian spectrum of a small classifier, bulk plus C outliers.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import hessian
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
OUT = Path(__file__).resolve().parent
torch.manual_seed(0)
torch.set_num_threads(4)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- style (same as da/lectures/assets/L2)
PAPER, INK, INK2, RULE = "#fcfbf7", "#1a1f26", "#4d5763", "#ddd7c9"
BLUE, ORANGE, GREEN, PURPLE, RED = "#1b6aa8", "#c0701a", "#0a8a5f", "#8a4fb0", "#b8442a"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
    "legend.fontsize": 15, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}  ({time.time() - T0:.0f} s)")


# ================================================================ valley.png
# A long valley: L = (S x^2 + s y^2)/2 with S/s = 400, plus a gentle slope so the floor is
# not exactly flat. GD at eta*S = 1.9 bounces across the walls while sliding along the floor.
S_wall, s_floor = 20.0, 0.4
eta_v = 1.98 / S_wall


def L_valley(x, y):
    return 0.5 * (S_wall * x**2 + s_floor * y**2)


p = np.array([0.6, 9.0]); traj = [p.copy()]
for _ in range(70):
    g = np.array([S_wall * p[0], s_floor * p[1]])
    p = p - eta_v * g; traj.append(p.copy())
traj = np.array(traj)
xs, ys = np.linspace(-1.2, 1.2, 300), np.linspace(-1, 10, 300)
XX, YY = np.meshgrid(xs, ys)
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.contour(XX, YY, L_valley(XX, YY), levels=np.geomspace(0.05, 20, 16), colors=INK2, linewidths=0.7, alpha=0.6)
ax.plot(traj[:, 0], traj[:, 1], "-", color=BLUE, lw=1.2, alpha=0.8)
ax.plot(traj[:, 0], traj[:, 1], "o", color=BLUE, ms=3.5)
ax.plot(traj[0, 0], traj[0, 1], "o", color=RED, ms=8)
ax.annotate("bounce across the walls:\nthe top eigenvector $u$,\n$\\eta S\\approx2$", xy=(-0.47, 4.2),
            xytext=(-1.17, 0.6), fontsize=13, arrowprops=dict(arrowstyle="->", color=INK2))
ax.annotate("drift along the floor:\nwhere progress lives", xy=(0.12, 2.6), xytext=(0.55, 4.6),
            fontsize=13, arrowprops=dict(arrowstyle="->", color=INK2))
ax.set_xlabel("sharp direction"); ax.set_ylabel("flat direction"); ax.set_aspect(0.22)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.8, 9.8); ax.grid(False)
ax.set_title(f"gradient descent at $\\eta S$ = {eta_v * S_wall:.1f}, curvature ratio {S_wall / s_floor:.0f}", fontsize=16)
save(fig, "valley.png")


# ================================================================ igr.png (td2_1 §1.2, code after Ex. 1.2.d)
def rk4(Fn, y0, T, n=4000):
    h = T / n; y = np.array(y0, float)
    for _ in range(n):
        k1 = Fn(y); k2 = Fn(y + h * k1 / 2); k3 = Fn(y + h * k2 / 2); k4 = Fn(y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def grad_ab(p):
    a, b = p; r = a * b - 1
    return np.array([r * b, r * a])


def hess_ab(p):
    a, b = p
    return np.array([[b * b, 2 * a * b - 1], [2 * a * b - 1, a * a]])


def gd_ab(p, eta, K):
    tr = [p]
    for _ in range(K):
        p = p - eta * grad_ab(p); tr.append(p)
    return np.array(tr)


p0, T_long = np.array([2.0, 0.1]), 40.0
F_gf = lambda p: -grad_ab(p)
F_igr = lambda eta: (lambda p: -(grad_ab(p) + 0.5 * eta * hess_ab(p) @ grad_ab(p)))
S_of = lambda p: np.linalg.eigvalsh(hess_ab(p))[-1]
p_gf = rk4(F_gf, p0, T_long, 8000)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
t_gf = np.linspace(0, 8, 60)
axs[0].plot(t_gf, [S_of(rk4(F_gf, p0, t, 300)) for t in t_gf], color=INK, label="gradient flow")
axs[1].axhline(p0[0]**2 - p0[1]**2, color=INK, label="gradient flow (conserved)")
captured = {}
for eta, col in [(0.05, BLUE), (0.2, GREEN), (0.35, ORANGE)]:
    tr = gd_ab(p0, eta, int(T_long / eta))
    p_igr = rk4(F_igr(eta), p0, T_long, 8000)
    shift = np.linalg.norm(tr[-1] - p_gf)
    captured[eta] = np.dot(p_igr - p_gf, tr[-1] - p_gf) / shift**2
    ts = np.arange(len(tr)) * eta; keep = ts <= 8
    axs[0].plot(ts[keep], [S_of(q) for q in tr[keep]], color=col, label=f"GD, $\\eta$ = {eta}")
    axs[1].plot(ts, tr[:, 0]**2 - tr[:, 1]**2, color=col, label=f"GD, $\\eta$ = {eta}")
    axs[1].plot(ts, [None] * len(ts))
    # the IGR flow's imbalance, as a dashed line
    n_igr = 400; tt = np.linspace(0, T_long, n_igr)
    q = p0.copy(); D = []
    for i in range(n_igr):
        D.append(q[0]**2 - q[1]**2)
        if i < n_igr - 1:
            q = rk4(F_igr(eta), q, tt[1] - tt[0], 20)
    axs[1].plot(tt, D, "--", color=col, lw=1.2)
axs[0].set(xlabel="$t=k\\eta$", ylabel="sharpness $S$", title="sharpness along the run")
axs[1].set(xlabel="$t=k\\eta$", ylabel="imbalance $a^2-b^2$", title="GD shrinks the imbalance (dashed: IGR flow)")
axs[0].legend(loc="lower right", fontsize=13); axs[1].legend(fontsize=13)
save(fig, "igr.png")
print("IGR flow captures of GD's shift from the GF endpoint:",
      ", ".join(f"eta={e}: {c:.0%} (eta*S0={e * S_of(p0):.2f})" for e, c in captured.items()))


# ================================================================ eos.png (td2_2 §2.1)
n_data, d_in, width = 256, 8, 64
torch.manual_seed(0)
X = torch.randn(n_data, d_in); Y = torch.sign(torch.randn(n_data, 1))


def make_mlp(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1))


def mse_net(net):
    return 0.5 * ((net(X) - Y) ** 2).mean()


def sharpness(net, v=None, iters=20):
    params = list(net.parameters())
    grads = torch.autograd.grad(mse_net(net), params, create_graph=True)
    if v is None:
        v = [torch.randn_like(p) for p in params]
    for _ in range(iters):
        norm = torch.sqrt(sum((a * a).sum() for a in v)); v = [a / norm for a in v]
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(Hv, v)).item()
        v = [a.detach() for a in Hv]
    return lam, v


def gd_step(net, eta):
    loss = mse_net(net)
    grads = torch.autograd.grad(loss, list(net.parameters()))
    with torch.no_grad():
        for q, g in zip(net.parameters(), grads):
            q -= eta * g
    return loss.item()


def run_gd(eta, t_max, dt_measure=1.25, seed=0):
    net = make_mlp(seed); every = max(1, round(dt_measure / eta))
    v, ts, Ss, losses = None, [], [], []
    for k in range(round(t_max / eta) + 1):
        if k % every == 0:
            lam, v = sharpness(net, v, iters=30 if v is None else 6); ts.append(k * eta); Ss.append(lam)
        losses.append(gd_step(net, eta))
    return np.array(ts), np.array(Ss), np.array(losses)


runs = {eta: run_gd(eta, t_max=250) for eta in (0.05, 0.1, 0.2)}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for (eta, (ts, Ss, losses)), c in zip(runs.items(), [BLUE, GREEN, ORANGE]):
    ax[0].plot(ts, Ss, color=c, label=f"$\\eta$ = {eta}")
    ax[0].axhline(2 / eta, color=c, ls="--", lw=1.2)
    ax[1].semilogy(np.arange(len(losses)) * eta, losses, color=c, lw=1.2)
ax[0].set(xlabel="$t=k\\eta$", ylabel="sharpness $S=\\lambda_{\\max}(H)$", title="dashed: $2/\\eta$")
ax[1].set(xlabel="$t=k\\eta$", ylabel="training loss", title="loss keeps falling, on average")
ax[0].legend()
save(fig, "eos.png")
for eta, (ts, Ss, losses) in runs.items():
    hover = Ss[ts > 150]
    print(f"eta={eta}: final sharpness {Ss[-1]:.1f} (2/eta = {2 / eta:.0f}); mean eta*S over t>150: {eta * hover.mean():.2f}")


# ================================================================ catapult.png (td2_2 §2.7)
def catapult(f0, a0, n, eta=1.0, K=200):
    f, a = np.full(K + 1, np.nan), np.full(K + 1, np.nan); f[0], a[0] = f0, a0
    for k in range(K):
        b = eta**2 * f[k] ** 2 / n
        f[k + 1], a[k + 1] = f[k] * (1 - a[k] + b), a[k] + b * (a[k] - 4)
        if not np.isfinite(f[k + 1]) or abs(f[k + 1]) > 1e100:
            f[k + 1] = a[k + 1] = np.nan; break
    return f, a


n_cat = 1000
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a0, col, lab in ((1.5, BLUE, "lazy"), (3.0, ORANGE, "catapult"), (4.1, RED, "divergent")):
    f, a = catapult(1.0, a0, n_cat)
    ax[0].semilogy(0.5 * f**2, color=col, label=f"$\\eta\\lambda_0$ = {a0} ({lab})")
    ax[1].plot(a, color=col)
ax[0].axhline(0.5 * n_cat, color=INK2, ls=":", lw=1.2)
ax[0].set(xlabel="step $k$", ylabel="loss $f^2/2$", xlim=(0, 60), ylim=(1e-12, 1e12), title=f"loss (dotted: $n/2$, $n$ = {n_cat})")
ax[1].axhline(2, color=INK2, ls="--", lw=1.2); ax[1].axhline(4, color=INK2, ls="--", lw=1.2)
ax[1].set(xlabel="step $k$", ylabel="$\\eta\\lambda_k$", xlim=(0, 60), ylim=(0, 5), title="curvature: dashed at 2 and 4")
ax[0].legend(fontsize=13)
save(fig, "catapult.png")
f, a = catapult(1.0, 3.0, n_cat)
print(f"catapult at eta*lambda_0=3: peak loss {np.nanmax(0.5 * f**2):.3g}, final eta*lambda {a[~np.isnan(a)][-1]:.2f}")


# ================================================================ floor.png (td2_3 §3.3)
# SGD on a diagonal quadratic with additive Gaussian gradient noise of covariance Sigma/B.
# The exact expected excess loss iterates the discrete Lyapunov recursion; the simulated
# runs average 400 replicas.
d_q = 12
h = np.geomspace(0.02, 1.0, d_q)              # curvatures
sig2 = np.full(d_q, 1.0)                      # noise variance per mode (commutes with H)


def excess_exact(eta, B, K, x0=None, eta_sched=None):
    # E[excess loss] per step, from the exact second-moment recursion mode by mode.
    m2 = np.zeros(d_q) if x0 is None else x0**2
    out = []
    for k in range(K):
        e = eta if eta_sched is None else eta_sched(k)
        out.append(0.5 * np.sum(h * m2))
        m2 = (1 - e * h) ** 2 * m2 + e**2 * sig2 / B
    return np.array(out)


def sgd_sim(eta, B, K, R=400, x0=None, eta_sched=None, seed=0):
    g = np.random.default_rng(seed)
    x = np.zeros((R, d_q)) if x0 is None else np.tile(x0, (R, 1)).astype(float)
    out = []
    for k in range(K):
        e = eta if eta_sched is None else eta_sched(k)
        out.append(0.5 * np.mean(np.sum(h * x**2, axis=1)))
        x = x - e * (h * x + np.sqrt(sig2 / B) * g.standard_normal(x.shape))
    return np.array(out)


K = 3000
fig, ax0 = plt.subplots(figsize=(6.6, 4.4))
for (eta, B), col, lab in (((0.2, 8), BLUE, "$\\eta$=0.2, $B$=8"), ((0.4, 16), GREEN, "$\\eta$=0.4, $B$=16"),
                           ((0.8, 32), ORANGE, "$\\eta$=0.8, $B$=32"), ((0.4, 8), RED, "$\\eta$=0.4, $B$=8")):
    sim = sgd_sim(eta, B, K)
    ax0.plot(np.arange(K) * eta, sim, color=col, lw=1.3, label=lab)
    ax0.axhline(eta / (4 * B) * sig2.sum(), color=col, ls="--", lw=1)
ax0.set(xlabel="$t=k\\eta$", ylabel="excess loss (mean of 400 runs)", title="dashed: the SDE floor $\\frac{\\eta}{4B}\\mathrm{tr}\\,\\Sigma$")
ax0.legend(fontsize=12, loc="lower right")
save(fig, "floor_scaling.png")
# learning-rate cut
k_cut = 1500
sched = lambda k: 0.4 if k < k_cut else 0.04
sim = sgd_sim(0.4, 8, K, eta_sched=sched); ex = excess_exact(0.4, 8, K, eta_sched=sched)
fig, ax1 = plt.subplots(figsize=(6.6, 4.4))
ax1.semilogy(sim, color=BLUE, lw=1.3, label="SGD, 400 runs")
ax1.semilogy(ex, color=INK, ls="--", lw=1.4, label="exact expectation")
ax1.axvline(k_cut, color=INK2, lw=1, ls=":")
ax1.set(xlabel="step $k$", ylabel="excess loss", title="$\\eta\\to\\eta/10$ at step 1500")
ax1.legend(fontsize=13)
save(fig, "floor_cut.png")
print(f"floor at eta=0.2,B=8: exact {excess_exact(0.2, 8, K)[-1]:.4f}, SDE {0.2 / 32 * sig2.sum():.4f}")


# ================================================================ balls.png (td2_4 §4.1)
g = np.array([1.0, 0.45]); g /= np.linalg.norm(g)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
th = np.linspace(0, 2 * np.pi, 400)
M = np.array([[1.0, 0.0], [0.0, 4.0]])             # metric for the ellipse: ||d||_M^2 = d^T M d
balls = [
    ("Euclidean: $-g$", np.c_[np.cos(th), np.sin(th)], -g, BLUE),
    ("$\\ell_\\infty$: $-\\mathrm{sign}\\,g$", np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]], float), -np.sign(g), ORANGE),
    ("$M$-norm: $-M^{-1}g$", np.c_[np.cos(th), np.sin(th)] @ np.linalg.inv(np.linalg.cholesky(M)).T,
     -np.linalg.solve(M, g) / np.sqrt(g @ np.linalg.solve(M, g)), GREEN),
]
for ax, (title, pts, step, col) in zip(axs, balls):
    ax.plot(pts[:, 0], pts[:, 1], color=INK2, lw=1.5)
    ax.annotate("", xy=-1.25 * g, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
    ax.text(-1.25 * g[0] - 0.42, -1.25 * g[1] - 0.05, "$-g$", fontsize=16, color=INK2)
    # supporting line perpendicular to g through the step point
    perp = np.array([-g[1], g[0]]); c = step @ g
    ln = np.array([step + 1.6 * perp, step - 1.6 * perp]); ax.plot(ln[:, 0], ln[:, 1], ":", color=col, lw=1.3)
    ax.annotate("", xy=step, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
    ax.plot(*step, "o", color=col, ms=8)
    ax.set_title(title, fontsize=17); ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("the step is where a line perpendicular to $g$ first touches the unit ball", fontsize=16, y=1.02)
save(fig, "balls.png")


# ================================================================ signnorm.png (td2_4 §4.5.D)
from scipy.special import gammaln


def top_sv(Sm, iters=60):
    v = rng.standard_normal(Sm.shape[1]); v /= np.linalg.norm(v); s = 0.0
    for _ in range(iters):
        u = Sm @ v; u /= np.linalg.norm(u); v = Sm.T @ u; s = np.linalg.norm(v); v /= s
    return s


ns = np.array([64, 128, 256, 512, 1024, 2048])
limit = lambda r: np.exp(gammaln(r / 2) - gammaln((r + 1) / 2)) / np.sqrt(np.pi)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for r, col in ((1, BLUE), (4, GREEN), (16, PURPLE), ("dense", ORANGE)):
    norms = []
    for n in ns:
        G = rng.standard_normal((n, n)) if r == "dense" else rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
        norms.append(top_sv(np.sign(G).astype(np.float32)))
    norms = np.array(norms); slope = np.polyfit(np.log(ns), np.log(norms), 1)[0]
    ax.loglog(ns, norms, "o-", ms=5, color=col, label=(f"rank {r}" if r != "dense" else "dense") + f": slope {slope:.2f}")
    ref = limit(r) * ns if r != "dense" else 2 * np.sqrt(ns)
    ax.loglog(ns, ref, ":", color=col, lw=1)
ax.set_xlabel("width $n$"); ax.set_ylabel("$\\|\\mathrm{sign}(G)\\|_2$"); ax.legend(fontsize=13)
ax.set_title("low rank: $\\propto n$;  dense random: $\\propto\\sqrt{n}$", fontsize=16)
save(fig, "signnorm.png")


# ================================================================ wd.png (td2_5 §5.2)
d_toy, a_toy = 64, 1.0


def sgd_scale_invariant(eta, lam, sigma, T, r0=1.0, seed=0):
    eta, lam, sigma, r0 = (np.array(z, float) for z in np.broadcast_arrays(
        np.atleast_1d(eta), np.atleast_1d(lam), np.atleast_1d(sigma), np.atleast_1d(r0)))
    Bn = len(eta); g = np.random.default_rng(seed)
    w = g.standard_normal((Bn, d_toy)); w *= r0[:, None] / np.linalg.norm(w, axis=1, keepdims=True)
    R, COS = np.empty((T, Bn)), np.empty((T, Bn))
    for k in range(T):
        r = np.linalg.norm(w, axis=1, keepdims=True); u = w / r
        gu = a_toy * u + sigma[:, None] * g.standard_normal(w.shape)
        gu -= (gu * u).sum(1, keepdims=True) * u
        w_new = (1 - eta * lam)[:, None] * w - eta[:, None] * gu / r
        R[k] = r[:, 0]; COS[k] = (w * w_new).sum(1) / r[:, 0] / np.linalg.norm(w_new, axis=1)
        w = w_new
    return dict(r=R, cos=np.clip(COS, -1, 1))


eta_w = np.array([0.1, 0.1, 0.4, 0.05, 0.2]); lam_w = np.array([1e-2, 1e-2, 2.5e-3, 2e-2, 2e-2])
sig_w = np.array([0.3, 1.0, 1.0, 0.3, 0.1]); T_w = 8000
out = sgd_scale_invariant(eta_w, lam_w, sig_w, T_w); theta = np.arccos(out["cos"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); win = 400
cols = [BLUE, GREEN, ORANGE, PURPLE, RED]
for i in range(len(eta_w)):
    rms = np.sqrt(np.convolve(theta[:, i] ** 2, np.ones(win) / win, mode="valid"))
    ax[0].plot(np.arange(len(rms)) + win, rms, color=cols[i], lw=1.4,
               label=f"$\\eta$={eta_w[i]}, $\\lambda$={lam_w[i]}, $\\sigma$={sig_w[i]}")
    ax[1].plot(out["r"][:, i], color=cols[i], lw=1.4)
for el in np.unique(eta_w * lam_w):
    ax[0].axhline(np.arccos(1 - el), color=INK, ls="--", lw=1.2)
ax[0].set(xlabel="step", ylabel="RMS angle per step", title="dashed: $\\arccos(1-\\eta\\lambda)$")
ax[1].set(xlabel="step", ylabel="$\\|w\\|$", title="the norm relaxes on a $1/(4\\eta\\lambda)$ timescale")
ax[0].legend(fontsize=11)
save(fig, "wd.png")
burn = T_w // 4
for i in range(len(eta_w)):
    print(f"eta={eta_w[i]}, lam={lam_w[i]}, sigma={sig_w[i]}: RMS angle {np.sqrt((theta[burn:, i]**2).mean()):.4f}, "
          f"arccos(1-eta*lam) {np.arccos(1 - eta_w[i] * lam_w[i]):.4f}")


# ================================================================ spectrum.png (td2_6 §6.3)
C, d_s, h_s, n_per = 5, 20, 24, 100
gen = torch.Generator().manual_seed(0)
mu = 2.0 * torch.randn(C, d_s, generator=gen) / d_s**0.5
Xs = torch.cat([mu[c] + torch.randn(n_per, d_s, generator=gen) / d_s**0.5 for c in range(C)])
ys = torch.arange(C).repeat_interleave(n_per)
shapes = [(h_s, d_s), (h_s,), (h_s, h_s), (h_s,), (C, h_s), (C,)]
sizes = [int(np.prod(s)) for s in shapes]
P = sum(sizes)


def net_s(theta):
    W1, b1, W2, b2, W3, b3 = [q.reshape(s) for q, s in zip(torch.split(theta, sizes), shapes)]
    return F.gelu(F.gelu(Xs @ W1.T + b1) @ W2.T + b2) @ W3.T + b3


def loss_s(theta):
    return F.cross_entropy(net_s(theta), ys)


ps = [torch.randn(*s, generator=gen) * (2 / s[1]) ** 0.5 if len(s) == 2 else torch.zeros(*s) for s in shapes]
theta = torch.cat([q.flatten() for q in ps]).requires_grad_(True)
opt = torch.optim.Adam([theta], lr=3e-3)
for step in range(400):
    opt.zero_grad(); l = loss_s(theta); l.backward(); opt.step()
    if l.item() < 0.25:            # stop well before interpolation: cross-entropy shrinks the outliers at the end
        break
acc = (net_s(theta).argmax(1) == ys).float().mean().item()
Hs = hessian(loss_s)(theta.detach()).detach()
ev = torch.linalg.eigvalsh(Hs).flip(0).numpy()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(ev, bins=120, color=BLUE)
ax[0].set_yscale("log"); ax[0].set(xlabel="eigenvalue", ylabel="count (log)", title=f"all {P} eigenvalues (log count)")
ax[1].plot(np.arange(1, 31), ev[:30], "o", color=BLUE, ms=5)
ax[1].axvline(C - 0.5, color=INK2, ls="--", lw=1.2)
ax[1].set(xlabel="index", ylabel="eigenvalue", title=f"top 30: about $C$ = {C} outliers")
save(fig, "spectrum.png")
print(f"spectrum: P={P}, train loss {l.item():.3f}, acc {acc:.2f}, top eigenvalues {np.round(ev[:7], 3)}, "
      f"lambda_{C}/lambda_{C + 1} = {ev[C - 1] / ev[C]:.2f}, median |eig| {np.median(np.abs(ev)):.2e}")
print(f"done in {time.time() - T0:.0f} s")
