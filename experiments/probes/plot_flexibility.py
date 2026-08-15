"""Figure: does each objective's residue preference track crystallographic mobility?

Left  - flexibility of each objective's top-ranked residue across within-protein
        B-factor quartiles (mean +/- SEM over structures).
Right - per-protein Spearman(log B, that flexibility); the distribution of each
        objective's *response* to the mobility gradient.

Uses the argmax residue rather than the distribution mean E[flex]: BayesDesign's
conditionals are flatter (H = 2.35 vs 1.75 nats for ProteinMPNN), which pulls any
expectation toward the uniform value (0.427) and would inflate the effect.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346,
    "Q": 0.493, "E": 0.497, "G": 0.544, "H": 0.323, "I": 0.462,
    "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509,
    "S": 0.507, "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
SERIES = [
    ("aa", "flex_native", "Native sequence", "#2a78d6"),
    ("argmax_struct", "flex_mpnn", "ProteinMPNN", "#eb6834"),
    ("argmax_bayes", "flex_bayes", "BayesDesign", "#1baf7a"),
]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
QUARTILES = ["Q1\nrigid", "Q2", "Q3", "Q4\nflexible"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="residues399.csv")
    ap.add_argument("--out", default="flexibility_vs_bfactor.png")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["logB"] = np.log(d["bfac"])
    for src, col, _, _ in SERIES:
        d[col] = d[src].map(FLEX)
    d["Bq"] = d.groupby("pdb")["bfac"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
    )
    n_prot = d["pdb"].nunique()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    # --- left: expected flexibility across B-factor quartiles ------------------
    x = np.arange(4)
    for _, ecol, label, color in SERIES:
        # Per-protein quartile means, then mean +/- SEM across structures.
        per = d.groupby(["pdb", "Bq"])[ecol].mean().unstack()
        mu, sem = per.mean(), per.std(ddof=1) / np.sqrt(len(per))
        ax1.errorbar(x, mu, yerr=sem, color=color, lw=2, marker="o", ms=8,
                     capsize=3, elinewidth=1.5, zorder=3,
                     markeredgecolor=SURFACE, markeredgewidth=2)
        # Contrast relief: every series carries a visible direct label.
        ax1.annotate(label, (x[-1] + 0.08, mu.iloc[-1]), color=INK,
                     fontsize=10, va="center", ha="left")

    ax1.set_xticks(x, QUARTILES, color=INK_2)
    ax1.set_xlim(-0.25, 4.05)
    ax1.set_xlabel("within-protein B-factor quartile", color=INK_2, fontsize=10)
    ax1.set_ylabel("flexibility of top-ranked residue", color=INK, fontsize=10)
    ax1.set_title("Native and ProteinMPNN follow the mobility gradient;\nBayesDesign attenuates it (~44%)",
                  color=INK, fontsize=11, loc="left", pad=12)

    # --- right: per-protein slope (Spearman) -----------------------------------
    proteins = list(d.groupby("pdb"))
    data, colors, labels = [], [], []
    for _, ecol, label, color in SERIES:
        data.append([stats.spearmanr(g["logB"], g[ecol]).statistic for _, g in proteins])
        colors.append(color)
        labels.append(label)

    bp = ax2.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color=SURFACE, lw=2),
                     whiskerprops=dict(color=INK_2, lw=1), capprops=dict(color=INK_2, lw=1))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    ax2.axvline(0, color=INK_2, lw=1, zorder=0)
    ax2.set_yticks(range(1, 4), labels, color=INK)
    ax2.set_xlabel(r"per-protein Spearman $\rho$(log B, residue flexibility)",
                   color=INK_2, fontsize=10)
    ax2.set_title(f"Response to the gradient, per structure (n = {n_prot})",
                  color=INK, fontsize=11, loc="left", pad=12)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
    ax1.grid(axis="x", visible=False)
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
