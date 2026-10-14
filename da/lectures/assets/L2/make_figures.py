"""Figures for DA lecture L2, "Curvature at scale".

Run from anywhere:  python da/lectures/assets/L2/make_figures.py
CPU only (numpy, matplotlib, torch); about 15 s. Writes PNGs next to this file
and prints the numerical checks that the verification log quotes.

Figures
  fd_error.png        relative error of the central finite-difference Hv against
                      autodiff Hv, as a function of the step eps, in fp64, fp32 and
                      "bf16" (fp32 gradients rounded to bf16).
  ritz.png            Ritz values of Lanczos against iteration m, with and without
                      full reorthogonalization (ghost eigenvalues without).
  quadrature.png      one probe v: the spectral measure mu_v of H seen from v (its CDF)
                      against the m-point Gauss rule that Lanczos produces.
  slq.png             exact spectral density (eigh) against SLQ at m = 90, 30, 10.
  filter.png          the spectrum of the Gauss-Newton matrix G with the damped-IF
                      filter at two dampings (0.1 x mean eigenvalue; between bulk and outliers).
"""
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
OUT = Path(__file__).resolve().parent
torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- style
PAPER, INK, INK2, RULE = "#fcfbf7", "#1a1f26", "#4d5763", "#ddd7c9"
BLUE, ORANGE, GREEN, PURPLE = "#1b6aa8", "#c0701a", "#0a8a5f", "#8a4fb0"
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
    print(f"wrote {name}")


# ---------------------------------------------------------------- the model
# A small tanh MLP on C = 5 overlapping Gaussian classes in d = 10, trained by
# full-batch Adam for 300 steps on the mean cross-entropy (not to interpolation).
# NB: the figures use the MEAN loss; summed-loss (DA1 convention) eigenvalues are n = 1000
# times larger. Ratios, fractions and relative errors are unaffected.
C, d, h, n = 5, 10, 32, 1000
mu = 1.0 * torch.randn(C, d, dtype=torch.float64)
y = torch.arange(C).repeat_interleave(n // C)
X = mu[y] + 1.0 * torch.randn(n, d, dtype=torch.float64)
shapes = [(h, d), (h,), (h, h), (h,), (C, h), (C,)]
sizes = [int(np.prod(s)) for s in shapes]
P = sum(sizes)


def unflat(w):
    return [p.reshape(s) for p, s in zip(torch.split(w, sizes), shapes)]


def logits(w, X):
    W1, b1, W2, b2, W3, b3 = unflat(w)
    a = torch.tanh(X @ W1.T + b1)
    a = torch.tanh(a @ W2.T + b2)
    return a @ W3.T + b3


def loss(w, X=X, y=y):
    return torch.nn.functional.cross_entropy(logits(w, X), y)  # mean loss


w0 = torch.cat([torch.randn(s, dtype=torch.float64).flatten() / np.sqrt(s[-1]) if len(s) == 2
                else torch.zeros(s, dtype=torch.float64) for s in shapes])
w = w0.clone().requires_grad_(True)
opt = torch.optim.Adam([w], lr=3e-3)
for _ in range(300):
    opt.zero_grad()
    L = loss(w)
    L.backward()
    opt.step()
w = w.detach()
acc = (logits(w, X).argmax(1) == y).double().mean().item()
print(f"P = {P} parameters, n = {n}, C = {C}; train loss {loss(w).item():.3f}, accuracy {acc:.2f}")

grad = torch.func.grad(loss)


def hvp_autodiff(w, v):
    """Forward-over-reverse: Hv = d/dr grad L(w + r v) at r = 0 (a jvp of the gradient)."""
    return torch.func.jvp(grad, (w,), (v,))[1]


# Exact Hessian (P x P) by vmapping the HVP over the identity, then eigh.
t = time.time()
H = torch.func.vmap(lambda e: hvp_autodiff(w, e))(torch.eye(P, dtype=torch.float64))
H = 0.5 * (H + H.T)
lam, U = np.linalg.eigh(H.numpy())
print(f"exact Hessian and eigh: {time.time() - t:.1f}s; "
      f"lambda_min {lam[0]:.3g}, lambda_max {lam[-1]:.3g}; top 8: {np.round(lam[-8:], 3)}")
Hn = H.numpy()
# outliers: we mark the top C eigenvalues; "bulk edge" = the (C+1)-th largest
n_out = C
bulk_edge = lam[-n_out - 1]
print(f"top {n_out + 3} eigenvalues / lambda_max: {np.round(lam[-n_out - 3:][::-1] / lam[-1], 3)}; "
      f"gap lambda_C / lambda_(C+1) = {lam[-n_out] / lam[-n_out - 1]:.2f}; "
      f"fraction of |lambda| < 1% of lambda_max: {np.mean(np.abs(lam) < 0.01 * lam[-1]):.3f}")

# ================================================================ Figure 1: FD error vs eps
v = torch.randn(P, dtype=torch.float64)
v /= v.norm()
Hv_exact = hvp_autodiff(w, v)
w32, v32 = w.float(), v.float()
X32 = X.float()


def grad32(wp):
    wp = wp.clone().requires_grad_(True)
    return torch.autograd.grad(loss(wp, X32, y), wp)[0]


eps_grid = np.logspace(-8, 0.5, 52)
err = {"fp64": [], "fp32": [], "bf16": []}
for eps in eps_grid:
    g_p, g_m = grad(w + eps * v), grad(w - eps * v)
    err["fp64"].append(((g_p - g_m) / (2 * eps) - Hv_exact).norm().item() / Hv_exact.norm().item())
    g_p, g_m = grad32(w32 + eps * v32), grad32(w32 - eps * v32)
    fd = ((g_p - g_m) / (2 * eps)).double()
    err["fp32"].append((fd - Hv_exact).norm().item() / Hv_exact.norm().item())
    g_p, g_m = g_p.bfloat16().float(), g_m.bfloat16().float()  # emulate bf16 gradients
    fd = ((g_p - g_m) / (2 * eps)).double()
    err["bf16"].append((fd - Hv_exact).norm().item() / Hv_exact.norm().item())
err = {k: np.array(e) for k, e in err.items()}
ad32 = (torch.func.jvp(torch.func.grad(lambda ww: loss(ww, X32, y)), (w32,), (v32,))[1].double()
        - Hv_exact).norm().item() / Hv_exact.norm().item()
print(f"autodiff Hv in fp32: relative error {ad32:.1e}")
for k in err:
    i = int(np.argmin(err[k]))
    print(f"FD Hv {k}: best eps {eps_grid[i]:.1e}, relative error {err[k][i]:.1e}")
# slopes: truncation branch (large eps) ~ eps^2, round-off branch (small eps) ~ 1/eps
sl_tr = np.polyfit(np.log(eps_grid[(eps_grid > 3e-3) & (eps_grid < 1e-1)]),
                   np.log(err["fp64"][(eps_grid > 3e-3) & (eps_grid < 1e-1)]), 1)[0]
ro = (eps_grid > 1e-6) & (eps_grid < 1e-4)
sl_ro = np.polyfit(np.log(eps_grid[ro]), np.log(err["fp32"][ro]), 1)[0]
print(f"log-log slope, truncation branch (fp64): {sl_tr:.2f} (expect 2); "
      f"round-off branch (fp32): {sl_ro:.2f} (expect -1)")

fig, ax = plt.subplots(figsize=(9.6, 5.0))
ax.loglog(eps_grid, err["fp64"], color=INK2, ls=":", label="fp64")
ax.loglog(eps_grid, err["fp32"], color=BLUE, label="fp32")
ax.loglog(eps_grid, err["bf16"], color=ORANGE, ls="--", label="bf16 gradients")
ax.axhline(ad32, color=BLUE, lw=1, alpha=0.6)
ax.text(1.3e-8, ad32 * 2.0, "autodiff $Hv$ in fp32", color=INK2, fontsize=13)
for k, c in [("fp32", BLUE), ("bf16", ORANGE)]:
    i = int(np.argmin(err[k]))
    ax.plot(eps_grid[i], err[k][i], "o", color=c, ms=9, mec=PAPER, mew=2)
ax.text(4e-8, 3e-4, r"round-off $\propto \epsilon_{\rm mach}/\epsilon$", color=INK2, fontsize=13)
ax.text(8e-2, 1e-6, r"truncation $\propto\epsilon^2$", color=INK2, fontsize=13)
ax.set_xlabel(r"finite-difference step $\epsilon$  ($\|v\|=1$)")
ax.set_ylabel(r"relative error of $Hv$")
ax.set_ylim(1e-9, 1e2)
ax.legend(loc="upper right")
save(fig, "fd_error.png")

# ================================================================ Lanczos and Gauss quadrature


def lanczos(matvec, q1, m, reorth=True):
    """m steps of Lanczos from unit q1. Returns alpha (m,), beta (m-1,), Q (P, m)."""
    Q = np.zeros((len(q1), m))
    alpha, beta = np.zeros(m), np.zeros(m)
    q, q_prev, b = q1 / np.linalg.norm(q1), np.zeros_like(q1), 0.0
    for k in range(m):
        Q[:, k] = q
        r = matvec(q) - b * q_prev
        alpha[k] = q @ r
        r -= alpha[k] * q
        if reorth:  # full reorthogonalization against every earlier basis vector, twice
            for _ in range(2):
                r -= Q[:, :k + 1] @ (Q[:, :k + 1].T @ r)
        b = np.linalg.norm(r)
        beta[k] = b
        if b < 1e-12:
            return alpha[:k + 1], beta[:k], Q[:, :k + 1]
        q_prev, q = q, r / b
    return alpha, beta[:m - 1], Q


def gauss_rule(alpha, beta):
    """Golub-Welsch: nodes = eigenvalues of T, weights = squared first components."""
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    theta, Y = np.linalg.eigh(T)
    return theta, Y[0] ** 2


matvec = lambda x: Hn @ x  # the only access to H that Lanczos uses

# check: three-term recurrence and Q^T H Q = T
a_, b_, Q_ = lanczos(matvec, rng.standard_normal(P), 30)
T_ = np.diag(a_) + np.diag(b_, 1) + np.diag(b_, -1)
print(f"Lanczos m=30: |Q^T H Q - T|_max = {np.abs(Q_.T @ Hn @ Q_ - T_).max():.1e}, "
      f"|Q^T Q - I|_max = {np.abs(Q_.T @ Q_ - np.eye(30)).max():.1e}")

# ================================================================ Figure 2: Ritz values vs m
M_RITZ = 60
q1 = rng.standard_normal(P)
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
ghost_report = {}
for ax, reorth, title in [(axs[0], True, "full reorthogonalization"),
                          (axs[1], False, "no reorthogonalization")]:
    a_, b_, _ = lanczos(matvec, q1, M_RITZ, reorth=reorth)
    for m in range(1, M_RITZ + 1):
        th = np.linalg.eigvalsh(np.diag(a_[:m]) + np.diag(b_[:m - 1], 1) + np.diag(b_[:m - 1], -1))
        ax.plot(np.full_like(th, m), th, "o", ms=3.2, color=BLUE if reorth else ORANGE, mew=0)
    th = np.linalg.eigvalsh(np.diag(a_) + np.diag(b_, 1) + np.diag(b_, -1))
    # count copies of the top eigenvalue
    ghost_report[title] = int(np.sum(np.abs(th - lam[-1]) < 1e-6 * lam[-1]))
    for e in lam[-n_out:]:
        ax.axhline(e, color=INK2, lw=0.7, ls=":")
    ax.axhline(lam[0], color=INK2, lw=0.7, ls=":")
    ax.set_title(title)
    ax.set_xlabel("Lanczos iteration $m$")
axs[0].set_ylabel(r"Ritz values (eigenvalues of $T_m$)")
print(f"copies of lambda_max among the Ritz values at m={M_RITZ}: {ghost_report}")
fig.tight_layout()
save(fig, "ritz.png")

# ================================================================ Figure 3: quadrature for one probe
m_q = 10
vq = rng.standard_normal(P)
vq /= np.linalg.norm(vq)
a_, b_, _ = lanczos(matvec, vq, m_q)
theta, tau2 = gauss_rule(a_, b_)
wts = (U.T @ vq) ** 2  # mu_v: nodes lam_j, weights (v . u_j)^2
# exactness for polynomials of degree <= 2m-1 (moments of mu_v)
mom_err = max(abs(np.sum(tau2 * theta ** k) - np.sum(wts * lam ** k)) / max(abs(np.sum(wts * lam ** k)), 1e-300)
              for k in range(2 * m_q))
mom_2m = abs(np.sum(tau2 * theta ** (2 * m_q)) - np.sum(wts * lam ** (2 * m_q))) / np.sum(wts * lam ** (2 * m_q))
print(f"Gauss rule m={m_q}: max relative moment error for degree 0..{2 * m_q - 1}: {mom_err:.1e}; "
      f"degree {2 * m_q}: {mom_2m:.1e}")
fig, ax = plt.subplots(figsize=(9.6, 4.8))
ax.step(lam, np.cumsum(wts), where="post", color=INK2, lw=1.4,
        label=rf"$\mu_v$: {P} nodes $\lambda_j$, weights $(v\cdot u_j)^2$")
ax.step(np.r_[lam[0], theta, lam[-1]], np.r_[0, np.cumsum(tau2), 1], where="post", color=BLUE,
        label=rf"Gauss rule from $T_{{{m_q}}}$: nodes $\theta_k$, weights $\tau_k^2$")
ax.plot(theta, np.cumsum(tau2), "o", color=BLUE, ms=7, mec=PAPER, mew=1.5)
ax.set_xscale("symlog", linthresh=0.05)
ax.set_xlabel(r"eigenvalue $s$ of $H$ (symlog axis)")
ax.set_ylabel("cumulative weight")
ax.legend(loc="lower right")
save(fig, "quadrature.png")

# ================================================================ Figure 4: SLQ vs exact
grid = np.linspace(lam[0] - 0.05 * (lam[-1] - lam[0]), lam[-1] + 0.05 * (lam[-1] - lam[0]), 3000)
SIG = 0.01 * (lam[-1] - lam[0])


def smooth(nodes, weights):
    z = (grid[:, None] - nodes[None, :]) / SIG
    return (np.exp(-0.5 * z ** 2) @ weights) / (np.sqrt(2 * np.pi) * SIG)


rho_exact = smooth(lam, np.full(P, 1.0 / P))


def slq(m, n_v, seed):
    r = np.random.default_rng(seed)
    nodes, weights = [], []
    for _ in range(n_v):
        a_, b_, _ = lanczos(matvec, r.choice([-1.0, 1.0], P), m)  # Rademacher probe
        th, t2 = gauss_rule(a_, b_)
        nodes.append(th)
        weights.append(t2 / n_v)
    return np.concatenate(nodes), np.concatenate(weights)


t = time.time()
slq_runs = {m: slq(m, 10, 1) for m in (90, 30, 10)}
print(f"SLQ runs: {time.time() - t:.1f}s")
for m, (nd, wt) in slq_runs.items():
    msg = []
    for k in (1, 2, 4):
        ex = np.mean(lam ** k)
        msg.append(f"m{k}: {np.sum(wt * nd ** k) / ex - 1:+.2%}")
    top = nd[wt > 0].max()
    msg.append(f"max node {top:.4g} vs lambda_max {lam[-1]:.4g}")
    msg.append(f"mass |lambda|<{SIG:.2g}: SLQ {wt[np.abs(nd) < SIG].sum():.3f} exact {np.mean(np.abs(lam) < SIG):.3f}")
    print(f"SLQ m={m}, n_v=10: " + "; ".join(msg))

fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
for ax, ms_ in [(axs[0], (90, 30)), (axs[1], (10,))]:
    ax.semilogy(grid, rho_exact, color=INK, lw=2.6, label="exact (eigh)")
    for m, c, ls in [(90, BLUE, "-"), (30, GREEN, "--"), (10, PURPLE, "-")]:
        if m in ms_:
            ax.semilogy(grid, smooth(*slq_runs[m]), color=c, ls=ls, lw=1.8, label=f"SLQ $m={m}$, $n_v=10$")
    for e in lam[-n_out:]:
        ax.axvline(e, color=ORANGE, lw=0.8, ls=":")
    ax.set_ylim(1e-5, rho_exact.max() * 3)
    ax.set_xlabel(r"eigenvalue $s$ of $H$")
    ax.legend(loc="upper right")
axs[0].set_ylabel(r"density $\rho(s)$, Gaussian-smoothed")
axs[0].set_title("enough iterations: bulk and outliers")
axs[1].set_title("$m=10$: outliers right, bulk coarse")
fig.tight_layout()
save(fig, "slq.png")

# ================================================================ Figure 5: spectrum of G and two dampings
# The IF uses the Gauss-Newton matrix G = (1/n) sum_i J_i^T H_y,i J_i (mean loss here), which is PSD.
Jac = torch.func.jacrev(logits)(w, X)  # (n, C, P)
pr = torch.softmax(logits(w, X), 1)
Hy = torch.diag_embed(pr) - pr[:, :, None] * pr[:, None, :]
G = torch.einsum("ncp,ncd,ndq->pq", Jac, Hy, Jac) / n
sG = np.linalg.eigvalsh(G.numpy())
print(f"G: top {C + 2} eigenvalues {np.round(sG[-C - 2:][::-1], 3)} (H: {np.round(lam[-C - 2:][::-1], 3)}); "
      f"|H-G|_F/|H|_F = {np.linalg.norm(Hn - G.numpy()) / np.linalg.norm(Hn):.2f}")
lam_mean = 0.1 * sG.mean()  # Grosse et al.'s rule, applied to the exact G
lam_gap = float(np.sqrt(sG[-C - 1] * sG[-C]))  # between the (C+1)-th and C-th largest
for name, ld in [("0.1 x mean eigenvalue", lam_mean), ("between bulk edge and outliers", lam_gap)]:
    print(f"damping {name}: lambda = {ld:.3g}; fraction of eigenvalues of G below it: {np.mean(sG < ld):.4f}")
pos = sG[sG > 1e-9]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
bins = np.logspace(np.log10(pos.min()), np.log10(sG[-1] * 1.5), 60)
ax.hist(pos, bins=bins, color=BLUE, alpha=0.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("count")
ax.set_title(f"eigenvalues of $G$ ({len(pos)} of {P} above $10^{{-9}}$)", fontsize=17)
ss = np.logspace(np.log10(pos.min()), np.log10(sG[-1] * 1.5), 400)
for ld, c, ls, lab in [(lam_mean, ORANGE, "-", r"$\lambda=0.1\times$mean eigenvalue"),
                       (lam_gap, PURPLE, "--", r"$\lambda$ between bulk and outliers")]:
    ax2.semilogx(ss, ld / (ss + ld), color=c, ls=ls, lw=2.4, label=lab)
    ax.axvline(ld, color=c, ls=ls, lw=1.2)
ax2.set_ylabel(r"$\lambda F(s)=\frac{\lambda}{s+\lambda}$")
ax2.set_ylim(0, 1.1)
ax2.set_xlabel(r"eigenvalue $s$")
ax2.legend(loc="lower left")
fig.tight_layout()
save(fig, "filter.png")

print(f"total runtime {time.time() - T0:.1f}s")
