"""Is the anti-naturalness specific to catalysis, or does any mutation cause it?

Evolved Kemp eliminases score lower under a sequence model than their parents, and
the mutated positions are strongly disfavoured. But mutations accumulate for many
reasons - stability, expression, drift - so "evolved variants look less natural"
may be trivially true of any mutated protein.

Two controls, both from data already in hand:

1. SPATIAL. Split each variant's mutations into those inside the active site and
   those outside. If anti-naturalness is the price of catalysis it should be
   concentrated at the site; if it is generic drift it should be uniform. (The
   ProteinMPNN version of this test gave a 30x concentration at the site in KE07.)

2. RANDOM-MUTATION NULL. For each variant, draw the same number of mutations at
   random positions, substituting to residues of matched background frequency, and
   score them. This gives the anti-naturalness expected from mutation alone. The
   real mutations must be *more* disfavoured than this null for the claim to hold.
"""

import argparse
import json
import sys

import numpy as np
from scipy import stats

# active-site residues, author numbering, per lineage (from ligand contacts)
SITE = {"HG3": [44, 50, 84, 127, 265], "KE07": [9, 11, 48, 49, 50, 101, 128, 201, 222],
        "KE59": []}
PARENT = {"HG3.3b": "HG3", "HG3.7": "HG3", "HG3.14": "HG3", "HG3.17": "HG3",
          "HG3.R5": "HG3", "HG3.17 Q50wt": "HG3", "Q50F": "HG3", "Q50M": "HG3",
          "Q50H": "HG3", "Q50S": "HG3", "Q50A": "HG3", "Q50K": "HG3",
          "KE07 r7": "KE07 r0", "KE59 R13": "KE59 R1"}
LINEAGE = {"HG3": "HG3", "KE07": "KE07", "KE59": "KE59"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="kemp_kinetics.json")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    xlnet = XLNetWrapper(device=torch.device(a.device))
    rows = {r["label"]: r for r in json.load(open(a.json))}
    rng = np.random.default_rng(0)

    def score(seq, positions):
        """log p at the given positions only (one forward batch per chunk)."""
        out = {}
        order = list(range(len(seq)))
        pos = sorted(positions)
        for s in range(0, len(pos), a.batch):
            toks = pos[s : s + a.batch]
            p = xlnet(seq=[seq] * len(toks), decode_order=order,
                      token_to_decode=toks, mask_type="bidirectional_mlm")
            p = p.float().cpu().numpy()
            for k, t in enumerate(toks):
                out[t] = float(np.log(p[k, aa20.index(seq[t])] + 1e-12))
        return out

    # background frequency, for matched random substitutions
    allseq = "".join(r["sequence"] for r in rows.values())
    freq = np.array([allseq.count(c) for c in aa20], float)
    freq /= freq.sum()

    results = []
    for child, parent in PARENT.items():
        if child not in rows or parent not in rows:
            continue
        cs, ps = rows[child]["sequence"], rows[parent]["sequence"]
        n = min(len(cs), len(ps))
        mut = [i for i in range(n) if cs[i] != ps[i]]
        if not mut:
            continue
        lin = "HG3" if child.startswith(("HG3", "Q50")) else child.split()[0]
        site = set(SITE.get(lin, []))
        real = score(cs, mut)
        d_real = float(np.mean(list(real.values())))

        # null: same count of random substitutions on the parent background
        nulls = []
        for _ in range(a.draws):
            idx = rng.choice(n, size=len(mut), replace=False)
            sub = list(ps)
            for i in idx:
                c = aa20[rng.choice(20, p=freq)]
                while c == sub[i]:
                    c = aa20[rng.choice(20, p=freq)]
                sub[i] = c
            sc = score("".join(sub), list(idx))
            nulls.append(float(np.mean(list(sc.values()))))
        z = (d_real - np.mean(nulls)) / (np.std(nulls) + 1e-9)
        inside = [real[i] for i in mut if (i + 1) in site]
        outside = [real[i] for i in mut if (i + 1) not in site]
        results.append({"variant": child, "n_mut": len(mut), "logp_real": d_real,
                        "logp_null": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
                        "z_vs_null": float(z),
                        "site": float(np.mean(inside)) if inside else np.nan,
                        "rest": float(np.mean(outside)) if outside else np.nan,
                        "n_site": len(inside)})
        r = results[-1]
        print(f"{child:<14} n_mut={r['n_mut']:>3}  real={d_real:+.3f}  "
              f"null={r['logp_null']:+.3f}+/-{r['null_sd']:.3f}  z={z:+.2f}  "
              f"site={r['site']:+.3f}({r['n_site']}) rest={r['rest']:+.3f}", flush=True)

    json.dump(results, open("naturalness_control.json", "w"), indent=1)
    z = np.array([r["z_vs_null"] for r in results])
    print(f"\n{'=' * 70}\nAre the real mutations more disfavoured than random ones?")
    print(f"  mean z vs random-mutation null = {z.mean():+.2f}  "
          f"({int((z < 0).sum())}/{len(z)} more disfavoured than random)")
    print("  z < 0 means evolution picked positions the model dislikes MORE than chance.")
    ok = [r for r in results if np.isfinite(r["site"]) and np.isfinite(r["rest"])]
    if ok:
        d = np.array([r["site"] - r["rest"] for r in ok])
        print(f"\n  active site vs rest, within the real mutations: "
              f"{d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}  (n={len(d)})")
        print("  negative = anti-naturalness concentrated at the active site.")


if __name__ == "__main__":
    main()
