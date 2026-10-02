"""
06_figures.py -- Figures 1-3 of the manuscript from ABO3_labels_and_descriptors.csv (05).
  Fig. 1  accuracy of the majority baseline and three geometric rules (both labels, four sets)
  Fig. 2  labelled compositions in the (t, mu) plane, coloured by structural topology (label L1)
  Fig. 3  perovskite fraction and topology composition by tolerance-factor bin
Each figure is written as vector PDF (embedded TrueType fonts), TIFF (1000 dpi) and PNG (600 dpi).
Needs pandas, numpy, matplotlib. Colours: Okabe-Ito colour-blind-safe palette.
"""

from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F"}
TOPO = [("perovskite", "Perovskite", OI["blue"]),
        ("non-perovskite (edge-sharing BO6)", "Edge-sharing BO$_6$", OI["verm"]),
        ("non-perovskite (face-sharing BO6)", "Face-sharing BO$_6$", OI["orange"]),
        ("non-perovskite (B not octahedral)", "B not octahedral", OI["green"]),
        ("non-perovskite (corner-sharing, A outside cavity)", "A outside cavity", OI["purple"]),
        ("non-perovskite (A under-coordinated)", "A under-coordinated", OI["sky"])]
EDGES = [0, 0.825, 0.90, 1.00, 1.059, 1.10 + 1e-9]
BINLAB = ["<0.825", "0.825–\n0.90", "0.90–\n1.00", "1.00–\n1.059", "1.059–\n1.10"]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{name}.png", dpi=600)
    plt.close(fig)


def load():
    d = pd.read_csv("ABO3_labels_and_descriptors.csv", sep=";", index_col=0)
    d = d.dropna(subset=["bartel_t", "bartel_mu", "bartel_tau"]).copy()
    d["conv"] = d.bartel_t.between(0.80, 1.10) & d.bartel_mu.between(0.414, 0.732)
    d["tb"] = (d.bartel_t > 0.825) & (d.bartel_t < 1.059)
    d["tau"] = d.bartel_tau < 4.18
    return d


def fig_accuracy(d):
    sets = [("All", d), ("No\noxoanion", d[~d.oxy]), ("Out-of-\nsample", d[~d.in_bartel]),
            ("Out-of-sample,\nno oxoanion", d[~d.in_bartel & ~d.oxy])]
    rules = [("Conventional window", "conv", OI["green"], "s"), ("Bartel $t$ bounds", "tb", OI["orange"], "^"),
             ("τ < 4.18", "tau", OI["blue"], "o")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, L, letter, title in zip(axes, ["L1", "L2"], ["a", "b"],
                                    ["L1: lowest-energy ICSD-matched structure", "L2: any ICSD-matched perovskite"]):
        for i, (_, s) in enumerate(sets):
            tr = s[L].astype(bool)
            k = max(int(tr.sum()), len(tr) - int(tr.sum()))
            p, lo, hi = wilson(k, len(s))
            ax.fill_between([i - 0.36, i + 0.36], 100 * lo, 100 * hi, color="0.88", lw=0, zorder=0)
            ax.plot([i - 0.36, i + 0.36], [100 * p] * 2, color="0.35", lw=1.2, zorder=1,
                    label="Majority baseline (95% CI)" if i == 0 else None)
            for j, (rname, col, colr, mk) in enumerate(rules):
                p, lo, hi = wilson(int((s[col] == tr).sum()), len(s))
                x = i + (j - 1) * 0.22
                ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=mk, ms=4.2,
                            mfc=colr, mec="black", mew=0.4, color=colr, capsize=2, lw=1.0, zorder=3,
                            label=rname if i == 0 else None)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([f"{n}\n(n = {len(s)})" for n, s in sets], fontsize=6.8)
        ax.set_xlim(-0.6, len(sets) - 0.4)
        ax.set_ylim(40, 100)
        ax.set_title(title, fontsize=7.5)
        ax.yaxis.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
        panel(ax, letter)
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Figure_1")


def fig_map(d):
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    ox = d[d.oxy]
    ax.scatter(ox.bartel_t, ox.bartel_mu, s=9, marker="o", facecolors="none", edgecolors="0.72", lw=0.5,
               label="Oxoanion-containing", zorder=1)
    nd = d[~d.oxy]
    for key, name, col in TOPO:
        q = nd[nd.L1_topology == key]
        if len(q):
            ax.scatter(q.bartel_t, q.bartel_mu, s=11, color=col, edgecolors="black", lw=0.25,
                       label=f"{name} ({len(q)})", zorder=3 if key == "perovskite" else 2)
    ax.add_patch(plt.Rectangle((0.80, 0.414), 0.30, 0.318, fill=False, lw=0.9, ls="--", ec="black", zorder=4))
    for x in (0.825, 1.059):
        ax.axvline(x, ls=":", lw=0.9, color="0.25", zorder=0)
    r = d.loc["RbNbO3"]
    ax.annotate("RbNbO$_3$, RbTaO$_3$", (r.bartel_t, r.bartel_mu), xytext=(1.075, 0.64), fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"))
    ax.set_xlim(0.62, 1.30)
    ax.set_ylim(0.10, 1.00)
    ax.set_xlabel("Tolerance factor $t$")
    ax.set_ylabel("Octahedral factor $\\mu$")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, handletextpad=0.2,
              columnspacing=0.8)
    fig.tight_layout()
    save(fig, "Figure_2")


def fig_bins(d):
    z = d[d.bartel_mu.between(0.414, 0.732)].copy()
    z["bin"] = pd.cut(z.bartel_t, EDGES, right=False, labels=BINLAB)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax in (a, b):
        ax.axvspan(0.5, 3.5, color="0.94", lw=0, zorder=0)
    for off, L, colr, mk in [(-0.12, "L1", OI["blue"], "o"), (0.12, "L2", OI["verm"], "s")]:
        for i, bl in enumerate(BINLAB):
            q = z[z.bin == bl]
            p, lo, hi = wilson(int(q[L].sum()), len(q))
            a.errorbar(i + off, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=mk, ms=4.2, mfc=colr,
                       mec="black", mew=0.4, color=colr, capsize=2, lw=1.0, label=L if i == 0 else None)
    a.set_ylim(0, 105)
    a.set_ylabel("Perovskite fraction (%)")
    a.legend(loc="upper left", fontsize=7)
    bottom = np.zeros(len(BINLAB))
    for key, name, col in TOPO:
        cnt = np.array([int((z[z.bin == bl].L1_topology == key).sum()) for bl in BINLAB])
        b.bar(range(len(BINLAB)), cnt, bottom=bottom, color=col, edgecolor="black", lw=0.3, width=0.65, label=name)
        bottom += cnt
    for i, n in enumerate(bottom):
        b.text(i, n + 2, f"{int(n)}", ha="center", fontsize=6.5)
    b.set_ylabel("Compositions (label L1)")
    b.set_ylim(0, max(bottom) * 1.15)
    b.legend(fontsize=6, loc="upper right", handlelength=1.2)
    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_xticks(range(len(BINLAB)))
        ax.set_xticklabels(BINLAB, fontsize=6.8)
        ax.set_xlim(-0.6, len(BINLAB) - 0.4)
        ax.set_xlabel("Tolerance factor $t$")
        panel(ax, letter)
    fig.tight_layout()
    save(fig, "Figure_3")


if __name__ == "__main__":
    data = load()
    fig_accuracy(data)
    fig_map(data)
    fig_bins(data)
    print("Figures 1-3 written (PDF, TIFF 1000 dpi, PNG 600 dpi)")
