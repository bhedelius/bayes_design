"""H1: is a better catalyst a more preorganized one?

Warshel's account of catalysis is that the enzyme's active site is already
arranged to stabilise the transition state, so no reorganization energy is paid.
In the model's terms, a preorganized site is one whose residue preferences barely
move when the ligand appears:

    reorg_i = KL[ p(a_i | struct, ligand) || p(a_i | struct, ligand masked) ]

Prediction: reorg at the active site DECREASES along a directed-evolution
trajectory, while the rest of the chain is unaffected.

Tested on ordered trajectories where every variant is crystallised with the SAME
ligand, so the sequence is the only thing changing.

Ligand handling matters: `get_ligand` sweeps up every HETATM in the file, so a
structure with more glycerol or more symmetry copies looks more "reorganized" for
purely crystallographic reasons (7 atoms for 6SRW vs 127 for 6TU6). Here the
ligand is filtered to the single copy of the target compound that is most buried
in the protein, and everything else is discarded.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from scipy import stats

# (series, pdb, variant label, activity rank; higher = better catalyst)
SERIES = [
    # Otten et al., all with 6NT bound, matched resolution: the primary ladder.
    ("HG3 lineage", "5RGA", "HG3", 0), ("HG3 lineage", "5RGB", "HG3.3b", 1),
    ("HG3 lineage", "5RGC", "HG3.7", 2), ("HG3 lineage", "5RGD", "HG3.14", 3),
    ("HG3 lineage", "5RGE", "HG3.17", 4),
    # Independent replicate of the same variants, higher resolution.
    ("HG3 replicate", "7K4Q", "HG3", 0), ("HG3 replicate", "7K4X", "HG3.7", 2),
    ("HG3 replicate", "7K4Z", "HG3.17", 4),
    # KE07, product-bound, by evolution round.
    ("KE07", "5D2T", "r0", 0), ("KE07", "5D2W", "r0", 0),
    ("KE07", "6DKV", "r5", 1), ("KE07", "6DC1", "r7", 2), ("KE07", "6CT3", "r7", 2),
    # KE59, TSA-bound.
    ("KE59", "3UXD", "R1", 0), ("KE59", "3UZ5", "R13", 1),
    # Single-position series on the HG3.17 background (activities not yet in hand,
    # so unranked): measures the metric's sensitivity to one substitution.
    ("Q50 series", "4BS0", "Q50(wt)", -1), ("Q50 series", "6SRW", "Q50F", -1),
    ("Q50 series", "6SRY", "Q50S", -1), ("Q50 series", "6SRZ", "Q50H", -1),
    ("Q50 series", "6SS1", "Q50A", -1), ("Q50 series", "6SS3", "Q50K", -1),
    ("Q50 series", "6TU6", "Q50M", -1),
    # Independent designed lineages, single points: breadth of the site signal.
    ("other designs", "5RGF", "HG4", -1), ("other designs", "8RD5", "HG3.R5", -1),
    ("other designs", "7K4U", "HG3 K50Q", -1), ("other designs", "3NYD", "HG3 orig", -1),
    ("other designs", "3NZ1", "1A53-2", -1), ("other designs", "3Q2D", "KE70", -1),
    ("other designs", "8FOS", "HG3-shell", -1), ("other designs", "8FOQ", "1A53-core", -1),
]
LIGAND = {"5RGA": "6NT", "5RGB": "6NT", "5RGC": "6NT", "5RGD": "6NT", "5RGE": "6NT",
          "5RGF": "6NT", "8RD5": "6NT",
          "7K4Q": "6NT", "7K4X": "6NT", "7K4Z": "6NT", "7K4U": "6NT",
          "4BS0": "6NT", "6SRW": "6NT", "6SRY": "6NT", "6SRZ": "6NT",
          "6SS1": "6NT", "6SS3": "6NT", "6TU6": "6NT", "8FOS": "6NT",
          "5D2T": "6VP", "5D2W": "6VP", "6DKV": "6VP", "6DC1": "6VP", "6CT3": "6VP",
          "3UXD": "0CU", "3UZ5": "0CU",
          "3NYD": "3NY", "3NZ1": "3NY", "3Q2D": "3NY", "8FOQ": "3NY"}


def clean_pdb(pdb, comp, cache):
    """Write a PDB holding chain A plus the single most-buried copy of `comp`."""
    src = os.path.join(cache, f"{pdb}.pdb")
    if not os.path.exists(src):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", src)
    prot, copies = [], {}
    for ln in open(src):
        if ln.startswith("ATOM") and ln[21] == "A":
            prot.append(ln)
        elif ln.startswith("HETATM") and ln[17:20].strip() == comp:
            copies.setdefault((ln[21], ln[22:27]), []).append(ln)
    if not prot or not copies:
        raise ValueError(f"no chain A / no {comp}")
    pc = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in prot])
    best, best_n = None, -1
    for key, lines in copies.items():
        lc = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
        n = int((np.linalg.norm(pc[:, None] - lc[None], axis=-1) < 5.0).sum())
        if n > best_n:
            best, best_n = lines, n
    # Ligand atoms often carry altloc codes B/C, which LigandMPNN's parser drops
    # (it keeps only blank/A) - that silently deletes most of the ligand. Keep the
    # highest-occupancy copy of each atom name and blank the altloc column.
    byname = {}
    for ln in best:
        name = ln[12:16]
        occ = float(ln[54:60]) if ln[54:60].strip() else 1.0
        if name not in byname or occ > byname[name][0]:
            byname[name] = (occ, ln[:16] + " " + ln[17:])
    best = [v[1] for v in byname.values()]
    out = os.path.join(cache, f"{pdb}_clean.pdb")
    with open(out, "w") as f:
        f.writelines(prot)
        f.writelines(best)
        f.write("END\n")
    return f"{pdb}_clean", len(best), best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--params", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--site", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import MPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER, get_ligand, seq_struct_from_ligand
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    params = a.params or os.path.join(a.src, "bayes_design", "ligand_mpnn", "model_params")
    ckpt = os.path.join(params, "ligandmpnn_v_32_010_25.pt")
    os.makedirs(a.cache, exist_ok=True)

    rows = []
    for series, pdb, label, rank in SERIES:
        try:
            stem, n_lig, n_contact = clean_pdb(pdb, LIGAND[pdb], a.cache)
            ctx = get_ligand(stem, structures_dir=a.cache, chain_list=("A",))
            seq, struct = seq_struct_from_ligand(ctx)
            if any(c not in aa20 for c in seq):
                raise ValueError("non-canonical residue")
            y = np.asarray(ctx["Y"])
            holo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                               checkpoint_path=ckpt, context=ctx, use_ligand=True)
            apo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                              checkpoint_path=ckpt, context=ctx, use_ligand=False)
            p_h = per_position_probs(holo, seq, struct, True, a.batch)
            p_a = per_position_probs(apo, seq, struct, True, a.batch)
            del holo, apo
            torch.cuda.empty_cache()

            bb = np.asarray(struct)
            d = np.linalg.norm(bb[:, :, None, :] - y[None, None, :, :], axis=-1).min((1, 2))
            site = d < a.site
            nat = np.array([aa20.index(c) for c in seq])
            idx = np.arange(len(seq))
            kl = (p_h * (np.log(p_h + 1e-12) - np.log(p_a + 1e-12))).sum(1)
            relief = np.log(p_h[idx, nat] + 1e-12) - np.log(p_a[idx, nat] + 1e-12)
            rows.append({"series": series, "pdb": pdb, "variant": label, "rank": rank,
                         "L": len(seq), "n_lig": n_lig, "n_site": int(site.sum()),
                         "reorg_site": float(kl[site].mean()),
                         "reorg_rest": float(kl[~site].mean()),
                         "relief_site": float(relief[site].mean())})
            r = rows[-1]
            print(f"{series:<14} {pdb} {label:<8} lig={n_lig:>3}atoms site={r['n_site']:>3} "
                  f"reorg_site={r['reorg_site']:.4f} rest={r['reorg_rest']:.4f} "
                  f"relief={r['relief_site']:+.4f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{series:<14} {pdb} {label}: FAILED {type(e).__name__}: {e}", flush=True)

    json.dump(rows, open("preorganization_results.json", "w"), indent=1)

    print("\n" + "=" * 70)
    print("H1: does active-site reorganization fall as catalysis improves?")
    print("=" * 70)
    for s in dict.fromkeys(r["series"] for r in rows):
        g = [r for r in rows if r["series"] == s]
        if len(g) < 3:
            if len(g) == 2:
                lo, hi = sorted(g, key=lambda r: r["rank"])
                print(f"\n{s}: {lo['variant']}={lo['reorg_site']:.4f} -> "
                      f"{hi['variant']}={hi['reorg_site']:.4f}  "
                      f"{'PREDICTED' if hi['reorg_site'] < lo['reorg_site'] else 'against'}")
            continue
        rk = np.array([r["rank"] for r in g], float)
        v = np.array([r["reorg_site"] for r in g])
        vr = np.array([r["reorg_rest"] for r in g])
        rel = np.array([r["relief_site"] for r in g])
        print(f"\n{s}  (n={len(g)})")
        for r in sorted(g, key=lambda r: r["rank"]):
            print(f"   {r['variant']:<8} rank={r['rank']}  reorg_site={r['reorg_site']:.4f}  "
                  f"rest={r['reorg_rest']:.4f}  relief={r['relief_site']:+.4f}")
        print(f"   Spearman(rank, reorg_site) = {stats.spearmanr(rk, v).statistic:+.3f}"
              f"   [control: rest = {stats.spearmanr(rk, vr).statistic:+.3f}]"
              f"   [H2 relief = {stats.spearmanr(rk, rel).statistic:+.3f}]")


if __name__ == "__main__":
    main()
