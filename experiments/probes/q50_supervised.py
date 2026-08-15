"""Supervised test: does active-site complementarity separate good catalysts from bad?

The Gln50 series is the cleanest labelled data available for this question: one
position varied on the HG3.17 background, the same transition-state analog bound
in every structure, all at 0.95-1.45 A.

Kries et al. (ACS Catal. 2020) report that tightly packed active sites (Gln, Met,
Phe) "perform consistently better than water-filled oxyanion holes found with
other substitutions (His, Ser, Ala, Lys)", and that Q50H specifically loses
16-fold in kcat/KM.

Pre-registered prediction: a packed site's residue preference should depend
strongly on the ligand, a water-filled one should not, so

    KL(Gln, Met, Phe)  >  KL(His, Ser, Ala, Lys)

This is a genuine out-of-sample test of the complementarity metric: the labels
come from the literature, not from anything the metric was tuned on.
"""

import argparse
import json

import numpy as np
from scipy import stats

PACKED = {"Q50(wt)": "4BS0", "Q50M": "6TU6", "Q50F": "6SRW"}
WATERY = {"Q50H": "6SRZ", "Q50S": "6SRY", "Q50A": "6SS1", "Q50K": "6SS3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reorg_perresidue.json")
    ap.add_argument("--site", type=float, default=6.0)
    a = ap.parse_args()

    with open(a.json) as f:
        store = json.load(f)

    def site_kl(pdb):
        r = store[pdb]
        kl = np.array(r["kl"])
        d = np.array(r["d"])
        m = d < a.site
        return (float(kl[m].mean()), int(m.sum())) if m.sum() else (np.nan, 0)

    print("Gln50 series: same backbone, same ligand, one position varied\n")
    print(f"{'variant':>9} {'pdb':>6} {'group':>8} {'n_site':>7} {'site KL':>9}")
    rows = []
    for group, members in (("packed", PACKED), ("watery", WATERY)):
        for v, pdb in members.items():
            if pdb not in store:
                print(f"{v:>9} {pdb:>6} {group:>8}   MISSING from scored set")
                continue
            kl, n = site_kl(pdb)
            rows.append({"variant": v, "pdb": pdb, "group": group, "kl": kl, "n": n})
            print(f"{v:>9} {pdb:>6} {group:>8} {n:>7} {kl:>9.4f}")

    p = [r["kl"] for r in rows if r["group"] == "packed" and np.isfinite(r["kl"])]
    w = [r["kl"] for r in rows if r["group"] == "watery" and np.isfinite(r["kl"])]
    if len(p) < 2 or len(w) < 2:
        print("\ntoo few usable structures for a test")
        return
    u = stats.mannwhitneyu(p, w, alternative="greater")
    print(f"\n  packed (Gln/Met/Phe): mean {np.mean(p):.4f}   n={len(p)}  {np.round(p, 4)}")
    print(f"  watery (His/Ser/Ala/Lys): mean {np.mean(w):.4f}   n={len(w)}  {np.round(w, 4)}")
    print(f"\n  Mann-Whitney U (packed > watery, one-sided): p = {u.pvalue:.3f}")
    # With n=3 vs 4 the smallest attainable one-sided p is 1/35 = 0.029, so a
    # perfect separation is the best possible outcome here.
    order = sorted(rows, key=lambda r: -r["kl"])
    perfect = all(r["group"] == "packed" for r in order[: len(p)])
    print(f"  ranking: {' > '.join(r['variant'] for r in order)}")
    print(f"  perfect separation: {'YES' if perfect else 'no'}"
          f"   (best attainable p with n={len(p)} vs {len(w)} is "
          f"{1 / (np.math.comb(len(p) + len(w), len(p))):.3f})")


if __name__ == "__main__":
    main()
