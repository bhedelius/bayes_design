"""Can a sequence model see the difference between a bad and a good catalyst?

CatPred, UniKP and DLKcat all regress kinetic parameters on protein language model
embeddings plus a substrate fingerprint. So their ceiling is set by whether the
underlying sequence representation distinguishes the variants at all. Here the
substrate is identical across every variant, which means the SMILES half of those
models contributes a constant - all discriminative power must come from sequence.

HG3 and HG3.17 are 94.7% identical (17 substitutions) and differ 100-fold in k_cat.
If the language model assigns them near-identical likelihood, no regressor built on
its embeddings can rank them, and the k_cat predictors are homology interpolators.

Reports, per variant:
  logL      mean log p(residue) over the sequence under ProtXLNet
  logL_mut  the same restricted to positions that differ from the lineage parent
"""

import argparse
import json
import sys

import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="kemp_kinetics.json")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    xlnet = XLNetWrapper(device=torch.device(a.device))
    with open(a.json) as f:
        rows = json.load(f)

    # Lineage parents, to define which positions are "the mutations".
    PARENT = {"HG3": ["HG3.3b", "HG3.7", "HG3.14", "HG3.17", "HG3.R5", "HG3.17 Q50wt",
                      "Q50F", "Q50M", "Q50H", "Q50S", "Q50A", "Q50K"],
              "KE07 r0": ["KE07 r7"], "KE59 R1": ["KE59 R13"]}
    by_label = {r["label"]: r for r in rows}

    def score(seq):
        out = np.zeros(len(seq))
        order = list(range(len(seq)))
        for s in range(0, len(seq), a.batch):
            toks = list(range(s, min(s + a.batch, len(seq))))
            p = xlnet(seq=[seq] * len(toks), decode_order=order,
                      token_to_decode=toks, mask_type="bidirectional_mlm")
            p = p.float().cpu().numpy()
            for k, t in enumerate(toks):
                out[t] = np.log(p[k, aa20.index(seq[t])] + 1e-12)
        return out

    cache = {}
    for r in rows:
        if any(c not in aa20 for c in r["sequence"]):
            print(f"skip {r['label']}: non-canonical")
            continue
        cache[r["label"]] = score(r["sequence"])
        print(f"scored {r['label']:<14} L={len(r['sequence'])} "
              f"logL={cache[r['label']].mean():+.4f}", flush=True)

    for parent, kids in PARENT.items():
        if parent not in cache:
            continue
        ps = by_label[parent]["sequence"]
        for k in kids:
            if k not in cache:
                continue
            ks = by_label[k]["sequence"]
            n = min(len(ps), len(ks))
            mut = [i for i in range(n) if ps[i] != ks[i]]
            by_label[k]["n_mut"] = len(mut)
            by_label[k]["logL_mut"] = float(cache[k][mut].mean()) if mut else np.nan
            by_label[k]["dlogL_mut"] = (float((cache[k][mut] - cache[parent][mut]).mean())
                                        if mut else np.nan)
        by_label[parent]["n_mut"] = 0

    print("\n" + "=" * 78)
    print("Does ProtXLNet likelihood track catalytic efficiency?")
    print("=" * 78)
    print(f"{'variant':>14} {'kcat':>8} {'n_mut':>6} {'logL':>9} {'logL_mut':>10} {'dlogL_mut':>10}")
    for r in rows:
        if r["label"] not in cache:
            continue
        r["logL"] = float(cache[r["label"]].mean())
        print(f"{r['label']:>14} {str(r['kcat'] or '-'):>8} {str(r.get('n_mut', '-')):>6} "
              f"{r['logL']:>9.4f} {r.get('logL_mut', float('nan')):>10.4f} "
              f"{r.get('dlogL_mut', float('nan')):>10.4f}")

    meas = [r for r in rows if r.get("kcat") and r["label"] in cache]
    if len(meas) >= 3:
        x = np.log10([r["kcat"] for r in meas])
        y = np.array([r["logL"] for r in meas])
        print(f"\n  measured set (n={len(meas)}): "
              f"Pearson(log10 kcat, logL) = {np.corrcoef(x, y)[0, 1]:+.3f}")
        for r in meas:
            print(f"    {r['label']:<10} kcat={r['kcat']:>7.1f}  logL={r['logL']:+.4f}")
        sp = [r for r in rows if r.get("rank") is not None and r["label"] in cache
              and r["label"].startswith("HG3")]
        if len(sp) >= 3:
            rk = np.array([r["rank"] for r in sp], float)
            v = np.array([r["logL"] for r in sp])
            print(f"  HG3 lineage by rank (n={len(sp)}): "
                  f"Spearman = {stats.spearmanr(rk, v).statistic:+.3f}")
    json.dump(rows, open("plm_vs_kcat.json", "w"), indent=1)


if __name__ == "__main__":
    main()
