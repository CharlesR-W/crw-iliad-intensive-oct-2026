#!/usr/bin/env python3
"""Illustrations for slt4_applications (NB4). Run from slt/: python assets/figures/make_slt4_applications_figures.py

1. slt4_applications_map.svg       how each application hangs off an idea from NB1-NB3
2. slt4_applications_llc_curve.svg a schematic (not data) of an LLC-over-training curve with stages

Also used on the page, not drawn here (downloaded 2026-10-14, checked by eye):
slt4_applications_lw_free_energy_crossover.png, the "Free energy crossover" figure of Liam Carroll,
"DSLT 4. Phase Transitions in Neural Networks" (LessWrong, 2023),
https://www.lesswrong.com/posts/aKBAYN5LpaQMrPqMj/dslt-4-phase-transitions-in-neural-networks
image URL: https://res.cloudinary.com/lesswrong-2-0/image/upload/f_png/v1/mirroredImages/aKBAYN5LpaQMrPqMj/u9ngzvbeaj0svkukttg0
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "font.family": "DejaVu Sans"})
INK, MUTED, ACCENT, PALE, GREY = "#222222", "#666666", "#2b6cb0", "#e8f0fa", "#f2f2f2"


def box(ax, x, y, w, h, title, sub, face, edge, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=face, ec=edge, lw=1.2, ls="--" if dashed else "-"))
    ax.text(x + 0.015, y + h * 0.66, title, ha="left", va="center", fontsize=11.5, color=INK, weight="bold")
    ax.text(x + 0.015, y + h * 0.28, sub, ha="left", va="center", fontsize=10.5, color=MUTED)


def arrow(ax, p, q, dashed=False, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def day_map():
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(-0.01, 1.015); ax.set_ylim(-0.03, 1.0); ax.axis("off")
    # left column: the path of the day
    lx, lw, lh = 0.0, 0.43, 0.17
    chain = [
        (0.80, "NB1  Bayes as stat mech", r"$\partial_h\,\mathbb{E}_h[O]=-\beta\,\mathrm{Cov}(O,\Delta)$"),
        (0.50, "NB2  Free energy and volume", r"volume exponent $\lambda$;  $\bar F_n\approx\lambda\log n+\dots$"),
        (0.20, "NB3  Exponents and temperature", r"$\hat\lambda$ from SGLD on the local posterior"),
    ]
    for y, t, s in chain:
        box(ax, lx, y, lw, lh, t, s, PALE, ACCENT)
    for (y0, *_), (y1, *_) in zip(chain, chain[1:]):
        arrow(ax, (lx + lw / 2, y0), (lx + lw / 2, y1 + lh), color=ACCENT)
    # right column: the applications
    rx, rw, rh = 0.55, 0.45, 0.13
    apps = [
        (0.585, "4.1  Stagewise development", r"$\hat\lambda$ at each training checkpoint"),
        (0.415, "4.2  Susceptibilities", r"$-\mathrm{Cov}(O,\Delta L)$: a small shift in the data"),
        (0.245, "4.3  Bayesian influence functions", r"$-\beta\,\mathrm{Cov}(O,\ell_i)$: one training sample"),
        (0.075, "4.4  Compressibility", r"$\hat\lambda$ vs how far a network compresses"),
    ]
    src = (lx + lw, chain[2][0] + lh / 2)
    for y, t, s in apps:
        box(ax, rx, y, rw, rh, t, s, "white", INK)
        arrow(ax, src, (rx, y + rh / 2))
    # the aside, kept visually separate
    y5 = 0.835
    box(ax, rx, y5, rw, rh, "4.5  Programs and singularities", "loss geometry of an encoded program",
        GREY, MUTED, dashed=True)
    arrow(ax, (lx + lw, chain[1][0] + lh / 2), (rx, y5 + rh / 2), dashed=True, color=MUTED)
    ax.text(0.5, 0.0, "all four estimate with SGLD at temperature, as in NB3", ha="center", fontsize=10,
            color=MUTED, style="italic")
    fig.savefig(OUT / "slt4_applications_map.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "slt4_applications_map.png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def llc_curve():
    # x = log10(step). Five stages separated by plateaus of lambda-hat.
    x = np.linspace(1.5, 5.0, 800)
    centres = np.array([2.2, 2.95, 3.55, 4.1, 4.6])          # where lambda-hat changes fastest
    dlam = np.array([25, 22, -6, 9, 3])                       # signs as in the LM run: + + - + +
    dloss = np.array([2.3, 1.2, 0.2, 0.4, 0.3])               # loss falls in every stage, mostly early
    width = 0.07
    s = lambda c: 0.5 * (1 + np.tanh((x - c) / width))
    lam = 4 + sum(d * s(c) for d, c in zip(dlam, centres))
    loss = 5.2 - sum(d * s(c) for d, c in zip(dloss, centres))
    bounds = (centres[:-1] + centres[1:]) / 2                 # flat parts of lambda-hat

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.08})
    edges = np.concatenate([[x[0]], bounds, [x[-1]]])
    for a in (a1, a2):
        for k in range(5):
            if k % 2 == 0:
                a.axvspan(edges[k], edges[k + 1], color=GREY, lw=0)
        for b in bounds:
            a.axvline(b, color=MUTED, ls="--", lw=0.9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.set_yticks([])
    a1.plot(x, loss, color=INK, lw=1.8)
    a1.set_ylabel("test loss", color=INK)
    a2.plot(x, lam, color=ACCENT, lw=2.2)
    a2.set_ylabel(r"$\hat\lambda$", color=ACCENT, rotation=0, labelpad=12, fontsize=12)
    for k in range(5):
        a1.text((edges[k] + edges[k + 1]) / 2, 1.02, f"stage {k + 1}", ha="center", va="bottom",
                fontsize=9.5, color=MUTED, transform=a1.get_xaxis_transform())
    # mark the stage where lambda-hat falls
    a2.annotate("$\\hat\\lambda$ falls\nin this stage", xy=(centres[2], lam[np.searchsorted(x, centres[2])] - 1),
                xytext=(centres[2], 14), ha="center", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=9))
    a2.annotate("flat: stage boundary", xy=(bounds[0], lam[np.searchsorted(x, bounds[0])]),
                xytext=(1.55, 42), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=9))
    a2.set_xticks([])
    a2.set_xlabel("training step (log scale)")
    a2.text(1.0, -0.04, "schematic, not data", transform=a2.transAxes, ha="right", va="top", fontsize=10,
            color=MUTED, style="italic")
    fig.savefig(OUT / "slt4_applications_llc_curve.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "slt4_applications_llc_curve.png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    day_map()
    llc_curve()
    print("wrote", *sorted(p.name for p in OUT.glob("slt4_applications_*")))
