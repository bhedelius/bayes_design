"""Positive control and noise floor for the reorganization metric.

Before any hypothesis test, two things must be true of the measurement:

A. POSITIVE CONTROL — it must detect something known to be true. The active site
   is where the ligand is; reorganization there must exceed a size- AND
   burial-matched random patch. Burial matching matters: the site is buried, so an
   unmatched null would only be measuring "buried vs exposed".

B. NOISE FLOOR — crystals of the SAME variant with the same ligand must give the
   same answer. If crystal-to-crystal spread is comparable to the spread across
   variants, no trend across variants is interpretable.

Saves per-residue KL so the analysis can be redone without re-running models.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

# Replicate pairs: same variant, same ligand, independent crystals.
REPLICATES = [
    ("HG3", "5RGA", "7K4Q"),
    ("HG3.7", "5RGC", "7K4X"),
    ("HG3.17", "5RGE", "7K4Z"),
    ("KE07 r0", "5D2T", "5D2W"),
    ("KE07 r7", "6DC1", "6CT3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--params", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--site", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import MPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER, get_ligand, seq_struct_from_ligand
    from preorganization import LIGAND, SERIES, clean_pdb
    from score_conditionals import neighbor_count, per_position_probs, virtual_cb

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    params = a.params or os.path.join(a.src, "bayes_design", "ligand_mpnn", "model_params")
    ckpt = os.path.join(params, "ligandmpnn_v_32_010_25.pt")
    rng = np.random.default_rng(0)

    store = {}
    for series, pdb, label, rank in SERIES:
        try:
            stem, n_lig, _ = clean_pdb(pdb, LIGAND[pdb], a.cache)
            ctx = get_ligand(stem, structures_dir=a.cache, chain_list=("A",))
            seq, struct = seq_struct_from_ligand(ctx)
            if any(c not in aa20 for c in seq):
                raise ValueError("non-canonical")
            y = np.asarray(ctx["Y"])
            holo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                               checkpoint_path=ckpt, context=ctx, use_ligand=True)
            apo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                              checkpoint_path=ckpt, context=ctx, use_ligand=False)
            p_h = per_position_probs(holo, seq, struct, True, a.batch)
            p_a = per_position_probs(apo, seq, struct, True, a.batch)
            del holo, apo
            torch.cuda.empty_cache()
            kl = (p_h * (np.log(p_h + 1e-12) - np.log(p_a + 1e-12))).sum(1)
            bb = np.asarray(struct)
            d = np.linalg.norm(bb[:, :, None, :] - y[None, None, :, :], axis=-1).min((1, 2))
            nb = neighbor_count(virtual_cb(bb))
            store[pdb] = {"series": series, "variant": label, "rank": rank,
                          "kl": kl.tolist(), "d": d.tolist(), "nbrs": nb.tolist()}
            print(f"scored {pdb} {label} L={len(seq)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{pdb} FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(store, open("reorg_perresidue.json", "w"))

    print("\n" + "=" * 72)
    print("A. POSITIVE CONTROL: true active site vs burial-matched random patches")
    print("=" * 72)
    print(f"{'pdb':>6} {'variant':>9} {'n_site':>7} {'site KL':>9} {'null KL':>9} "
          f"{'z':>7} {'percentile':>11}")
    zs = []
    for pdb, r in store.items():
        kl = np.array(r["kl"])
        d = np.array(r["d"])
        nb = np.array(r["nbrs"])
        site = d < a.site
        n = int(site.sum())
        if n < 4:
            continue
        # Draw null patches with a burial distribution matched to the true site.
        others = np.where(~site)[0]
        tgt = np.sort(nb[site])
        null = []
        for _ in range(a.draws):
            pick = []
            pool = list(others)
            for t in tgt:
                j = min(pool, key=lambda i, t=t: abs(nb[i] - t))
                pick.append(j)
                pool.remove(j)
            # jitter the greedy match so draws differ
            rng.shuffle(pool)
            swap = rng.integers(0, len(pick), size=max(1, len(pick) // 3))
            for s_ in swap:
                cand = [i for i in pool if abs(nb[i] - nb[pick[s_]]) <= 2]
                if cand:
                    pick[s_] = cand[rng.integers(0, len(cand))]
            null.append(kl[pick].mean())
        null = np.array(null)
        z = (kl[site].mean() - null.mean()) / (null.std() + 1e-12)
        pct = float((null < kl[site].mean()).mean() * 100)
        zs.append(z)
        print(f"{pdb:>6} {r['variant']:>9} {n:>7} {kl[site].mean():>9.4f} "
              f"{null.mean():>9.4f} {z:>7.1f} {pct:>10.1f}%")
    print(f"\n  mean z across {len(zs)} structures = {np.mean(zs):.1f}   "
          f"min z = {np.min(zs):.1f}")
    print("  -> the metric DOES detect the site" if np.min(zs) > 2 else
          "  -> the metric does NOT reliably detect the site")

    print("\n" + "=" * 72)
    print("B. NOISE FLOOR: same variant, same ligand, different crystals")
    print("=" * 72)
    diffs = []
    for name, p1, p2 in REPLICATES:
        if p1 not in store or p2 not in store:
            continue
        v = []
        for p in (p1, p2):
            kl = np.array(store[p]["kl"])
            d = np.array(store[p]["d"])
            v.append(kl[d < a.site].mean())
        rel = abs(v[0] - v[1]) / np.mean(v) * 100
        diffs.append(rel)
        print(f"  {name:<9} {p1}={v[0]:.4f}  {p2}={v[1]:.4f}   "
              f"difference = {rel:.0f}% of the mean")
    across = [np.array(r["kl"])[np.array(r["d"]) < a.site].mean() for r in store.values()]
    print(f"\n  median within-variant (crystal) difference: {np.median(diffs):.0f}%")
    print(f"  spread ACROSS all variants: {np.std(across) / np.mean(across) * 100:.0f}% (CV)")
    print("\n  -> trends across variants are interpretable"
          if np.median(diffs) < np.std(across) / np.mean(across) * 100 / 2 else
          "\n  -> crystal noise is comparable to the across-variant signal:"
          "\n     trends across variants are NOT interpretable")


if __name__ == "__main__":
    main()
