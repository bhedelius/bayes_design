"""Does the BayesDesign objective read out crystallographic flexibility?

Primary test: per-protein Spearman correlation between B-factor and each model
score, aggregated over structures (Wilcoxon signed-rank vs 0, and paired between
scores). Secondary: the same correlation after partialling out burial, and
whether the residue each objective prefers is less flexible than the native one.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

# Bhaskaran & Ponnuswamy average flexibility index (the ProtParam scale).
FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346,
    "Q": 0.493, "E": 0.497, "G": 0.544, "H": 0.323, "I": 0.462,
    "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509,
    "S": 0.507, "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}

SCORES = ["logp_struct", "logp_seq", "logp_bayes", "logp_rawratio", "H_struct", "H_bayes"]
LABEL = {
    "logp_struct": "ProteinMPNN  log p(a|struct)",
    "logp_seq": "ProtXLNet    log p(a|seq)",
    "logp_bayes": "BayesDesign  log ratio (bal)",
    "logp_rawratio": "BayesDesign  log ratio (raw)",
    "H_struct": "ProteinMPNN  entropy",
    "H_bayes": "BayesDesign  entropy",
    "nbrs": "burial (CB neighbors)",
}


def partial_spearman(x, y, *covars):
    """Spearman of x,y after linearly removing the covariates from both (on ranks)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    z = np.c_[np.ones_like(rx), *[stats.rankdata(c) for c in covars]]
    ex = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    ey = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    return stats.spearmanr(ex, ey).statistic


def summarize(vals, name):
    vals = np.asarray(vals)
    p = stats.wilcoxon(vals).pvalue
    return f"{name:<32s} rho = {vals.mean():+.3f} +/- {vals.std(ddof=1) / np.sqrt(len(vals)):.3f}   p = {p:.2g}"


def main(path):
    d = pd.read_csv(path)
    d["logB"] = np.log(d["bfac"])
    # Termini are systematically high-B; keep it as a covariate.
    d["term"] = d.groupby("pdb")["pos"].transform(lambda p: np.minimum(p, p.max() - p))
    proteins = list(d.groupby("pdb"))
    scores = [s for s in SCORES if s in d.columns]
    print(f"{len(proteins)} structures, {len(d)} residues\n")

    print("=" * 78)
    print("1. Per-protein Spearman(B-factor, score)   mean +/- SEM over structures")
    print("=" * 78)
    raw, partial = {}, {}
    for s in scores + ["nbrs"]:
        raw[s] = [stats.spearmanr(g["logB"], g[s]).statistic for _, g in proteins]
        partial[s] = [partial_spearman(g["logB"], g[s], g["nbrs"], g["term"]) for _, g in proteins]
    for s in scores + ["nbrs"]:
        print(summarize(raw[s], LABEL[s]))

    print("\n" + "=" * 78)
    print("2. Same, after partialling out burial (CB neighbors) and distance to terminus")
    print("=" * 78)
    for s in scores:
        print(summarize(partial[s], LABEL[s]))

    print("\n" + "=" * 78)
    print("3. Paired comparison: is BayesDesign more B-factor-sensitive than MPNN?")
    print("=" * 78)
    for tag, table in (("raw", raw), ("burial-controlled", partial)):
        diff = np.array(table["logp_bayes"]) - np.array(table["logp_struct"])
        p = stats.wilcoxon(diff).pvalue
        n_neg = int((diff < 0).sum())
        print(
            f"{tag:<20s} rho(bayes) - rho(mpnn) = {diff.mean():+.3f} +/- "
            f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}   p = {p:.2g}   "
            f"({n_neg}/{len(diff)} proteins more negative)"
        )

    print("\n" + "=" * 78)
    print("4. Flexibility of the preferred residue (Bhaskaran-Ponnuswamy index)")
    print("=" * 78)
    d["flex_native"] = d["aa"].map(FLEX)
    d["flex_mpnn"] = d["argmax_struct"].map(FLEX)
    d["flex_bayes"] = d["argmax_bayes"].map(FLEX)
    d["Bq"] = d.groupby("pdb")["bfac"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    tbl = d.groupby("Bq")[["flex_native", "flex_mpnn", "flex_bayes"]].mean()
    tbl.index = ["Q1 (rigid)", "Q2", "Q3", "Q4 (flexible)"]
    print(tbl.round(4).to_string())
    print("\nGly fraction by B-factor quartile:")
    gly = d.assign(
        g_nat=(d.aa == "G"), g_mpnn=(d.argmax_struct == "G"), g_bayes=(d.argmax_bayes == "G")
    ).groupby("Bq")[["g_nat", "g_mpnn", "g_bayes"]].mean()
    gly.index = ["Q1 (rigid)", "Q2", "Q3", "Q4 (flexible)"]
    print(gly.round(4).to_string())

    # Paired over proteins: mean flexibility of preferred residue in the flexible quartile.
    q4 = d[d.Bq == 3].groupby("pdb")[["flex_native", "flex_mpnn", "flex_bayes"]].mean()
    print("\nFlexible quartile, paired over structures:")
    for a, b in (("flex_mpnn", "flex_native"), ("flex_bayes", "flex_native"), ("flex_bayes", "flex_mpnn")):
        diff = q4[a] - q4[b]
        print(
            f"  {a} - {b:<12s} = {diff.mean():+.4f} +/- {diff.std(ddof=1) / np.sqrt(len(diff)):.4f}"
            f"   p = {stats.wilcoxon(diff).pvalue:.2g}"
        )

    if "eflex_bayes" in d.columns:
        print("\n" + "=" * 78)
        print("4b. Does expected flexibility track B-factor?  per-protein Spearman(B, E[flex])")
        print("=" * 78)
        proteins = list(d.groupby("pdb"))  # regroup: flex_* were added after the first grouping
        cols = {"flex_native": "native sequence", "eflex_struct": "ProteinMPNN E[flex]",
                "eflex_seq": "ProtXLNet   E[flex]", "eflex_bayes": "BayesDesign E[flex]"}
        eflex = {c: [stats.spearmanr(g["logB"], g[c]).statistic for _, g in proteins] for c in cols}
        for c, name in cols.items():
            print(summarize(eflex[c], name))
        diff = np.array(eflex["eflex_bayes"]) - np.array(eflex["flex_native"])
        print(
            f"\n  BayesDesign - native slope = {diff.mean():+.3f} +/- "
            f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}   p = {stats.wilcoxon(diff).pvalue:.2g}"
        )
        diff = np.array(eflex["eflex_bayes"]) - np.array(eflex["eflex_struct"])
        print(
            f"  BayesDesign - MPNN slope   = {diff.mean():+.3f} +/- "
            f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}   p = {stats.wilcoxon(diff).pvalue:.2g}"
        )

    print("\n" + "=" * 78)
    print("5. Sequence recovery by B-factor quartile")
    print("=" * 78)
    rec = d.assign(r_mpnn=(d.argmax_struct == d.aa), r_bayes=(d.argmax_bayes == d.aa)).groupby("Bq")[
        ["r_mpnn", "r_bayes"]
    ].mean()
    rec.index = ["Q1 (rigid)", "Q2", "Q3", "Q4 (flexible)"]
    print(rec.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "residues.csv")
