#!/usr/bin/env python3
"""Conceptual illustrations for SLT Notebook 1 (slt1_bayes_statmech).

Run from slt/:  python assets/figures/make_slt1_bayes_statmech_figures.py
Writes assets/figures/slt1_bayes_statmech_<name>.svg (numpy + matplotlib only, a few seconds).
To look at them: rsvg-convert -z 1.5 <file>.svg -o /tmp/<file>.png

Figures (one per idea, placed where the idea is introduced):
  dictionary     prior and energy combine into the posterior; areas are partition functions (§1.1)
  concentration  posterior ellipses N(w_hat, (nH)^-1) shrinking by sqrt(10) per tenfold n (§1.2)
  covariances    w_hat scatter over data sets vs the posterior's covariance, well specified
                 vs misspecified, simulated for the model N(mu, e^{2s}) (§1.2)
  occam          the Occam factor as a ratio of two volumes, 1-D and 2-D (§1.3)
  predictors     Bayes / Gibbs / maximum-likelihood predictive densities in the misspecified
                 example of Exercise 1.4.4 (§1.4)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
STEM = "slt1_bayes_statmech"

INK, MUTED = "#222222", "#777777"
BLUE, ORANGE, GREEN, GREY = "#2f6ea8", "#d9822b", "#3a8f5c", "#b8b8b8"

plt.rcParams.update({
    "font.size": 11, "mathtext.fontset": "cm", "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "figure.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / f"{STEM}_{name}.svg")
    plt.close(fig)


def bare(ax, xlabel=None, ylabel=None):
    """Schematic axes: a baseline, no ticks."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def ellipse(ax, mean, cov, k=2.0, **kw):
    """The k-sigma contour of N(mean, cov)."""
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    e = Ellipse(mean, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang, **kw)
    ax.add_patch(e)
    return e


# ---------------------------------------------------------------------------- dictionary
def fig_dictionary():
    w = np.linspace(-3, 3, 1201)
    prior = gauss(w, 0.0, 1.4)
    # a likelihood with a deep narrow well (U, right) and a shallower wide one (V, left)
    energy = -np.logaddexp(-0.5 * ((w - 1.3) / 0.16) ** 2, np.log(0.45) - 0.5 * ((w + 1.1) / 0.5) ** 2)
    lik = np.exp(-energy)
    post = prior * lik
    cut = w[np.argmin(np.where(np.abs(w - 0.3) < 0.6, post, np.inf))]  # the barrier between the wells

    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.6), gridspec_kw={"wspace": 0.4})
    heads = [("prior $\\varphi(w)$", "weight on microstates"),
             ("energy $nL_n(w)$", "energy of microstate $w$"),
             ("weight $\\varphi\\,e^{-nL_n}$", "unnormalized Boltzmann weight")]
    for ax, (top, sub) in zip(axs, heads):
        bare(ax, "$w$")
        ax.set_title(top, fontsize=12, color=INK, pad=18)
        ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=MUTED, style="italic")

    axs[0].fill_between(w, prior, color=GREY, alpha=0.5, lw=0)
    axs[0].plot(w, prior, color=INK, lw=1.4)
    axs[0].set_ylim(0, prior.max() * 1.25)

    axs[1].plot(w, energy, color=INK, lw=1.4)
    axs[1].set_ylim(energy.min() - 0.6, energy.min() + 7)
    axs[1].text(1.3, energy.min() - 0.45, "deep, narrow", ha="center", fontsize=9, color=MUTED)
    axs[1].text(-1.1, energy[np.argmin(np.abs(w + 1.1))] - 0.45, "shallow, wide", ha="center",
                fontsize=9, color=MUTED)

    ax = axs[2]
    ax.plot(w, post, color=INK, lw=1.4)
    left, right = w <= cut, w >= cut
    ax.fill_between(w[left], post[left], color=ORANGE, alpha=0.45, lw=0)
    ax.fill_between(w[right], post[right], color=BLUE, alpha=0.45, lw=0)
    ax.plot([cut, cut], [0, post.max() * 0.9], color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, post.max() * 1.45)
    ax.text(-1.75, post[left].max() * 0.75, "$Z_n(V)$", ha="right", fontsize=11, color=INK)
    ax.text(1.55, post.max() * 0.8, "$Z_n(U)$", ha="left", fontsize=11, color=INK)
    ax.text(-2.9, post.max() * 1.42, "total area $Z_n$:\npartition function", fontsize=9, color=MUTED,
            va="top")
    ax.text(-2.2, -post.max() * 0.08, "$V$", ha="center", fontsize=10, color=ORANGE)
    ax.text(2.2, -post.max() * 0.08, "$U$", ha="center", fontsize=10, color=BLUE)

    for x0 in (0.355, 0.645):
        fig.text(x0, 0.45, "$\\times\\;e^{-(\\cdot)}$" if x0 < 0.5 else "$\\Rightarrow$",
                 ha="center", va="center", fontsize=13, color=MUTED)
    save(fig, "dictionary")


# ---------------------------------------------------------------------------- concentration
def fig_concentration():
    H = np.array([[2.4, 0.9], [0.9, 0.6]])        # anisotropic Hessian: one stiff, one soft direction
    w0 = np.array([0.35, 0.25])
    offsets = {5: np.array([-0.35, 0.45]), 50: np.array([0.10, -0.16]), 500: np.array([-0.03, 0.05])}
    shades = {5: 0.18, 50: 0.35, 500: 0.6}

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    g = np.linspace(-2.6, 2.6, 300)
    X, Y = np.meshgrid(g, g)
    ax.imshow(np.exp(-0.5 * (X**2 + Y**2) / 1.3**2), extent=(-2.6, 2.6, -2.6, 2.6), origin="lower",
              cmap="Greys", vmin=0, vmax=4.0, interpolation="bilinear")
    ax.text(-2.45, 2.25, "prior $\\varphi$", color=MUTED, fontsize=10)
    for n, off in offsets.items():
        c = w0 + off / 1.0
        ellipse(ax, c, np.linalg.inv(n * H), fc=BLUE, alpha=shades[n], ec=BLUE, lw=1.1)
        ax.plot(*c, ".", color=INK, ms=4)
    # labels beside each ellipse
    ax.annotate("$n=5$", w0 + offsets[5], xytext=(-2.1, -1.6), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.annotate("$n=50$", w0 + offsets[50], xytext=(1.25, -1.5), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.annotate("$n=500$", w0 + offsets[500], xytext=(1.3, 1.5), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.plot(*w0, "+", color=ORANGE, ms=10, mew=2.0, zorder=6)
    ax.annotate("$w_0$", w0, xytext=(w0[0] + 0.45, w0[1] - 0.55), color=ORANGE, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal")
    bare(ax, "$w_1$", "$w_2$")
    ax.spines["left"].set_visible(True)
    save(fig, "concentration")


# ---------------------------------------------------------------------------- covariances
def fig_covariances():
    """Model N(mu, e^{2s}); w0 = (true mean, log true sd) = (0, 0) for both data sources below."""
    rng = np.random.default_rng(1)
    n, reps = 400, 600
    H = np.diag([1.0, 2.0])                          # Hessian of L at w0 when Var(x) = 1
    srcs = [("well specified: data $\\mathcal{N}(0,1)$", lambda size: rng.normal(0, 1, size)),
            ("misspecified: skewed data $\\mathrm{Exp}(1)-1$", lambda size: rng.exponential(1, size) - 1)]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.4), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.12})
    for ax, (title, draw) in zip(axs, srcs):
        x = draw((reps, n))
        mu_hat = x.mean(1)
        s_hat = 0.5 * np.log(x.var(1))
        pts = np.sqrt(n) * np.c_[mu_hat, s_hat]
        ax.plot(pts[:, 0], pts[:, 1], ".", color=INK, ms=2.2, alpha=0.55)
        ellipse(ax, (0, 0), np.linalg.inv(H), k=2.0, fc=BLUE, alpha=0.18, ec=BLUE, lw=1.6)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color=GREY, lw=0.6, zorder=0); ax.axvline(0, color=GREY, lw=0.6, zorder=0)
        ax.set_xlabel("$\\sqrt{n}\\,(\\hat\\mu-\\mu_0)$")
    axs[0].set_ylabel("$\\sqrt{n}\\,(\\hat s-s_0)$")
    axs[0].text(-4.0, 3.6, "dots: $\\hat w$ on 600 data sets", fontsize=9, color=INK)
    axs[0].text(-4.0, 3.0, "shaded: posterior's $2\\sigma$ ellipse", fontsize=9, color=BLUE)
    save(fig, "covariances")


# ---------------------------------------------------------------------------- occam
def fig_occam():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
    ax = axs[0]
    w = np.linspace(-2.6, 2.6, 1201)
    a, b, wh = -2.0, 2.0, 0.55
    ax.add_patch(Rectangle((a, 0), b - a, 1.0, fc=GREY, alpha=0.35, ec=MUTED, lw=1.0))
    s = 0.22
    bump = np.exp(-0.5 * ((w - wh) / s) ** 2)
    ax.fill_between(w, bump, color=BLUE, alpha=0.5, lw=0)
    ax.plot(w, bump, color=BLUE, lw=1.4)
    bump4 = np.exp(-0.5 * ((w - wh) / (s / 2)) ** 2)
    ax.plot(w, bump4, color=INK, lw=1.0, ls="--")
    ax.annotate("", (a, 1.12), (b, 1.12), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(0, 1.17, "prior volume $V_\\varphi$", ha="center", va="bottom", color=INK, fontsize=10)
    ax.annotate("posterior\nvolume $V_{\\rm post}$", (wh + 0.25, 0.55),
                xytext=(1.2, 0.68), color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.annotate("$4n$: half as wide", (wh - 0.12, 0.6), xytext=(-1.85, 0.62), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(-1.85, 0.22, "height 1 at $\\hat w$:\n$e^{-n(L_n(w)-L_n(\\hat w))}$", color=MUTED, fontsize=9)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(0, 1.4)
    bare(ax, "$w$")
    ax.set_title("$d=1$: Occam factor $=V_{\\rm post}/V_\\varphi$", fontsize=11, color=INK)

    ax = axs[1]
    ax.add_patch(Rectangle((-1, -1), 2, 2, fc=GREY, alpha=0.35, ec=MUTED, lw=1.0))
    ellipse(ax, (0.25, 0.15), np.array([[0.012, 0.006], [0.006, 0.006]]), k=2.0,
            fc=BLUE, alpha=0.6, ec=BLUE, lw=1.0)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("$d=2$: one factor $\\propto n^{-1/2}$ per axis", fontsize=11, color=INK)
    ax.text(0.25, -0.35, "posterior contour\n(schematic)", ha="center", fontsize=9, color=BLUE)
    save(fig, "occam")


# ---------------------------------------------------------------------------- predictors
def fig_predictors():
    """Exercise 1.4.4: model N(x; w, 1), flat prior, data q = N(0, 2), n = 4 (small, so the effect shows)."""
    n, sigma2, xbar, w_gibbs = 4, 2.0, 0.4, 1.05
    v = 1.0 / n
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.12})

    ax = axs[0]
    w = np.linspace(-1.6, 2.4, 800)
    post = gauss(w, xbar, np.sqrt(v))
    ax.fill_between(w, post, color=BLUE, alpha=0.25, lw=0)
    ax.plot(w, post, color=BLUE, lw=1.4)
    # six equal-weight posterior members at the quantiles (k - 1/2)/6
    members = xbar + np.sqrt(v) * np.array([-1.383, -0.674, -0.210, 0.210, 0.674, 1.383])
    for m in members:
        ax.plot([m, m], [0, gauss(m, xbar, np.sqrt(v))], color=BLUE, lw=0.7, alpha=0.7)
    ax.plot([xbar, xbar], [0, post.max()], color=INK, lw=1.8)
    ax.plot([w_gibbs, w_gibbs], [0, gauss(w_gibbs, xbar, np.sqrt(v))], color=ORANGE, lw=1.8)
    ax.text(xbar, post.max() * 1.04, "$\\hat w$", ha="center", color=INK)
    ax.text(w_gibbs + 0.05, gauss(w_gibbs, xbar, np.sqrt(v)) + 0.03, "$w\\sim\\pi_n$", color=ORANGE)
    ax.set_ylim(0, post.max() * 1.2)
    bare(ax, "$w$")
    ax.set_title("posterior $\\pi_n$ over the parameter", fontsize=11, color=INK)

    ax = axs[1]
    x = np.linspace(-4.0, 4.5, 1000)
    ax.fill_between(x, gauss(x, 0, np.sqrt(sigma2)), color=GREY, alpha=0.4, lw=0)
    for m in members:
        ax.plot(x, gauss(x, m, 1.0), color=BLUE, lw=0.6, alpha=0.45)
    ax.plot(x, gauss(x, xbar, 1.0), color=INK, lw=1.6, label="maximum likelihood $p(x\\mid\\hat w)$")
    ax.plot(x, gauss(x, w_gibbs, 1.0), color=ORANGE, lw=1.6, ls="--", label="Gibbs $p(x\\mid w)$, $w\\sim\\pi_n$")
    ax.plot(x, gauss(x, xbar, np.sqrt(1 + v)), color=BLUE, lw=2.4,
            label="Bayes $p^*(x)$: posterior average")
    ax.plot([], [], color=BLUE, lw=0.6, alpha=0.6, label="thin: $p(x\\mid w)$, posterior members")
    ax.text(-3.9, 0.06, "data $q$", color=MUTED, fontsize=10)
    ax.set_ylim(0, 0.52)
    bare(ax, "$x$")
    fig.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("predictive densities over the data", fontsize=11, color=INK)
    save(fig, "predictors")


if __name__ == "__main__":
    for f in (fig_dictionary, fig_concentration, fig_covariances, fig_occam, fig_predictors):
        f()
    print("wrote", sorted(p.name for p in OUT.glob(f"{STEM}_*.svg")))
