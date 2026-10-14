#!/usr/bin/env python3
"""Conceptual figures for SLT Notebook 3 (slt3_effective_temperature).

Run from slt/:  python assets/figures/make_slt3_effective_temperature_figures.py
Writes four SVGs next to this script. numpy + matplotlib only; a few seconds.

  _jump.svg        lambda(delta) jumps at delta = 0, lambda_eff(n; delta) is smooth;
                   at delta = 0 the sample loss has a random quadratic term ~ n^{-1/2}.
  _two_scales.svg  K = w^4 + delta w^2 at a coarse and a fine loss threshold t, and the
                   log-log picture: slope 4 above t ~ delta^2, slope 2 below (local volume exponent = 1/slope in 1-D).
  _wbic.svg        E_beta[n L_n] ~ n L_n(w*) + lambda / beta is linear in 1/beta with slope lambda;
                   it meets F_n ~ n L_n(w*) + lambda log n at 1/beta = log n.
  _sgld.svg        Localized Langevin chains on K = (w1 w2)^2 from w* = 0, loose and tight tether gamma.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
STEM = "slt3_effective_temperature"

INK = "#1f2328"
GREY = "#9aa0a8"
BLUE = "#2f6db5"
ORANGE = "#d0782a"
SHADE = "#cfe0f3"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "mathtext.fontset": "cm",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{STEM}_{name}.svg", bbox_inches="tight", pad_inches=0.05, dpi=160)
    plt.close(fig)


# lambda_eff for K = w^4 + delta w^2 with a flat prior on R: a function of s = delta sqrt(n) only,
# lambda_eff = n E[K] under exp(-nK) = E[u^4 + s u^2] under exp(-u^4 - s u^2).
_u = np.linspace(-5, 5, 40_001)


def lam_eff_s(s):
    e = _u**4 + s * _u**2
    p = np.exp(-e)
    return (p * e).sum() / p.sum()


# ------------------------------------------------------------------ figure 1: the jump
def fig_jump():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.32})

    a = ax[0]
    deltas = np.linspace(0, 0.2, 401)
    for n, shade in [(1e2, 0.75), (1e3, 0.55), (1e4, 0.35), (1e5, 0.15)]:
        lam = [lam_eff_s(d * np.sqrt(n)) for d in deltas]
        c = str(shade)
        a.plot(deltas, lam, color=c, lw=1.2)
        lab_y = {1e2: 0.36, 1e3: 0.42, 1e4: 0.465, 1e5: 0.485}[n]
        k = np.argmin(np.abs(np.array(lam) - lab_y))
        a.text(deltas[k] + 0.004, lam[k] - 0.012, rf"$n=10^{{{int(np.log10(n))}}}$", fontsize=10, color=c, va="top")
    a.plot([0.0015, 0.2], [0.5, 0.5], color=BLUE, lw=2.2)
    a.plot([0], [0.5], "o", mfc="white", mec=BLUE, ms=6, mew=1.6, zorder=5)
    a.plot([0], [0.25], "o", color=BLUE, ms=6, zorder=5)
    a.text(0.198, 0.515, r"$\lambda=\frac{1}{2}$ for $\delta>0$", color=BLUE, ha="right", va="bottom")
    a.text(0.008, 0.25, r"$\lambda=\frac{1}{4}$ at $\delta=0$", color=BLUE, va="center")
    a.set_xlim(-0.006, 0.205)
    a.set_ylim(0.2, 0.58)
    a.set_xticks([0, 0.1, 0.2])
    a.set_yticks([0.25, 0.5])
    a.set_yticklabels([r"$\frac{1}{4}$", r"$\frac{1}{2}$"], fontsize=12)
    a.set_xlabel(r"$\delta$")
    a.set_title(r"(a) $\lambda$ jumps; $\lambda_{\rm eff}(n)$ (grey) does not", fontsize=11.5)

    b = ax[1]
    # Exercise 3.1.4 at delta = 0 (a_n set to 1): K_n = w^4/2 - u_n w^2, with u_n ~ N(0, sigma^2/n).
    # Drawn at u_n = 0, +-1 and +-2 standard deviations rather than at random draws.
    n, sigma = 1000, 1.0
    sd = sigma / np.sqrt(n)
    w = np.linspace(-0.3, 0.3, 801)
    for k in [-2, -1, 1, 2]:
        b.plot(w, 0.5 * w**4 - k * sd * w**2, color=ORANGE, lw=1.2)
    b.plot(w, 0.5 * w**4, color=INK, lw=2, ls="--")
    b.text(0.0, -0.0013, "double wells", color=ORANGE, fontsize=10, ha="center", va="top")
    wc = n ** -0.25
    b.annotate("", xy=(-wc, -0.0029), xytext=(wc, -0.0029), arrowprops=dict(arrowstyle="<->", color=GREY, lw=1))
    b.text(0, -0.0032, r"$|w|\lesssim n^{-1/4}$", ha="center", va="top", color="#61666d", fontsize=10.5)
    b.axhline(0, color=GREY, lw=0.6)
    b.set_xlim(-0.3, 0.3)
    b.set_ylim(-0.0045, 0.0045)
    b.set_xticks([])
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel(r"$w$")
    b.set_title(r"(b) $\delta=0$: $K=\frac{1}{2}w^4$ (dashed), samples $K_n$", fontsize=11.5)
    save(fig, "jump")


# ------------------------------------------------------------------ figure 2: two scales
def fig_two_scales():
    delta = 0.01
    fig = plt.figure(figsize=(8.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.35)

    def panel(ax, wmax, t, ref, ref_label, title, note):
        w = np.linspace(-wmax, wmax, 801)
        K = w**4 + delta * w**2
        inside = K < t
        ax.fill_between(w, K, t, where=inside, color=SHADE, lw=0)
        ax.plot(w, K, color=INK, lw=2.4)
        ax.plot(w, ref(w), color=ORANGE, lw=1.4, ls="--")
        ax.axhline(t, color=BLUE, lw=1, ls=":")
        ax.text(wmax * 0.98, t, r"$t$", color=BLUE, ha="right", va="bottom")
        wt = w[inside].max()
        ax.plot([-wt, wt], [0, 0], color=BLUE, lw=4, solid_capstyle="butt")
        ax.text(0, -0.07 * 1.6 * t, note, ha="center", va="top", color=BLUE, fontsize=10.5)
        ax.text(0, 1.5 * t, ref_label, color=ORANGE, fontsize=10.5, va="top", ha="center")
        ax.set_xlim(-wmax, wmax)
        ax.set_ylim(-0.32 * t, 1.6 * t)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.set_title(title, fontsize=11.5)

    t_coarse, t_fine = 0.1, 1e-5
    a = fig.add_subplot(gs[0])
    panel(a, 0.85, t_coarse, lambda w: w**4, r"dashed: $w^4$", r"(a) coarse: $t\gg\delta^2$", r"width $\propto t^{1/4}$")
    b = fig.add_subplot(gs[1])
    panel(b, 0.05, t_fine, lambda w: delta * w**2, r"dashed: $\delta w^2$", r"(b) fine: $t\ll\delta^2$", r"width $\propto t^{1/2}$")

    c = fig.add_subplot(gs[2])
    w = np.logspace(-3.2, 0, 400)
    c.loglog(w, w**4, color=GREY, lw=1.2, ls="--")
    c.loglog(w, delta * w**2, color=GREY, lw=1.2, ls="--")
    c.loglog(w, w**4 + delta * w**2, color=INK, lw=1.8)
    for t, lab in [(t_coarse, "(a)"), (t_fine, "(b)")]:
        c.axhline(t, color=BLUE, lw=1, ls=":")
        c.text(1.3e-3 * 0.62, t * 1.4, lab, color=BLUE, fontsize=10.5, va="bottom")
    c.axhline(delta**2, color=ORANGE, lw=0.8)
    c.text(0.62e-3, delta**2 * 1.4, r"$t\sim\delta^2$", color=ORANGE, fontsize=10.5, va="bottom")
    c.text(0.19, 0.05, "slope 4\n" + r"$V\propto t^{1/4}$", fontsize=10.5, ha="right", va="top")
    c.text(4e-3, 1e-8, "slope 2\n" + r"$V\propto t^{1/2}$", fontsize=10.5, ha="left", va="bottom")
    c.set_xlim(0.6e-3, 1)
    c.set_ylim(1e-10, 2)
    c.set_xlabel(r"$|w|$ (log)")
    c.set_ylabel(r"$K$ (log)")
    c.set_xticks([])
    c.set_yticks([])
    c.minorticks_off()
    c.set_title(r"(c) $\log K$ against $\log|w|$", fontsize=11.5)
    save(fig, "two_scales")


# ------------------------------------------------------------------ figure 3: WBIC
def fig_wbic():
    lam, logn, base = 1.0, 6.0, 1.0
    T = np.linspace(0, 7.5, 200)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(T, base + lam * T, color=INK, lw=2)
    ax.axhline(base, color=GREY, lw=1, ls=":")
    F = base + lam * logn
    ax.axhline(F, color=BLUE, lw=1.4, ls="--")
    ax.text(0.1, F + 0.15, r"free energy $F_n\approx nL_n(w^*)+\lambda\log n$", color=BLUE, va="bottom", fontsize=10.5)
    ax.text(4.3, base + lam * 4.3 - 0.35, r"$\mathbb{E}_\beta[nL_n]\approx nL_n(w^*)+\lambda/\beta$",
            ha="left", va="top", fontsize=10.5)
    ax.plot([logn], [F], "o", color=BLUE, ms=6, zorder=5)
    ax.plot([logn, logn], [0, F], color=BLUE, lw=0.8, ls=":")
    ax.text(logn + 0.1, F - 0.25, "WBIC", color=BLUE, fontsize=10.5, va="top")
    ax.plot([1], [base + lam], "o", color=INK, ms=5, zorder=5)
    ax.plot([1, 1], [0, base + lam], color=GREY, lw=0.8, ls=":")
    ax.text(1.12, base + lam - 0.1, r"Bayes, $\beta=1$", fontsize=10.5, va="top")
    # slope triangle
    x0, dx = 2.2, 1.4
    ax.plot([x0, x0 + dx, x0 + dx], [base + lam * x0, base + lam * x0, base + lam * (x0 + dx)], color=ORANGE, lw=1.2)
    ax.text(x0 + dx + 0.1, base + lam * (x0 + dx / 2), r"slope $\lambda$", color=ORANGE, fontsize=10.5, va="center")
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 8.6)
    ax.set_xticks([0, 1, logn])
    ax.set_xticklabels(["0", "1", r"$\log n$"])
    ax.set_yticks([base, base + lam, F])
    ax.set_yticklabels([r"$nL_n(w^*)$", r"$nL_n(w^*)+\lambda$", r"$F_n$"])
    ax.set_xlabel(r"temperature $1/\beta$")
    ax.set_ylabel(r"posterior mean of $nL_n$")
    save(fig, "wbic")


# ------------------------------------------------------------------ figure 4: localized SGLD
def fig_sgld():
    rng = np.random.default_rng(7)
    nbeta, eps, steps = 200.0, 2e-4, 60_000
    g = np.linspace(-1.7, 1.7, 401)
    W1, W2 = np.meshgrid(g, g)
    K = (W1 * W2) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for a, gamma, title in [(ax[0], 1.0, r"(a) loose tether, $\gamma=1$"), (ax[1], 25.0, r"(b) tight tether, $\gamma=25$")]:
        a.contourf(W1, W2, np.log10(K + 1e-6), levels=np.linspace(-4, 1, 11), cmap="Greys", alpha=0.45)
        a.contour(W1, W2, K, levels=[1 / nbeta], colors=[BLUE], linewidths=1.0)
        w = np.zeros(2)
        path = np.empty((steps, 2))
        for s in range(steps):
            grad = nbeta * 2 * np.array([w[0] * w[1] ** 2, w[1] * w[0] ** 2]) + gamma * w
            w = w - 0.5 * eps * grad + np.sqrt(eps) * rng.standard_normal(2)
            path[s] = w
        a.plot(path[::10, 0], path[::10, 1], color=ORANGE, lw=0.5, alpha=0.8, rasterized=True)
        r = gamma ** -0.5
        a.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", color=INK, lw=1.2))
        a.annotate(r"radius $\gamma^{-1/2}$", xy=(r * 0.707, -r * 0.707), xytext=(0.55, -1.35), fontsize=10.5,
                   arrowprops=dict(arrowstyle="-", color=INK, lw=0.8), bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        a.plot([0], [0], "o", color=INK, ms=5, zorder=5)
        a.text(0.08, 0.08, r"$w^*$", fontsize=11, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        a.set_xlim(-1.7, 1.7)
        a.set_ylim(-1.7, 1.7)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(GREY)
        a.set_xlabel(r"$w_1$")
        a.set_ylabel(r"$w_2$")
        a.set_title(title, fontsize=11.5)
    fig.tight_layout()
    save(fig, "sgld")


if __name__ == "__main__":
    fig_jump()
    fig_two_scales()
    fig_wbic()
    fig_sgld()
    print("wrote", sorted(p.name for p in OUT.glob(f"{STEM}_*.svg")))
