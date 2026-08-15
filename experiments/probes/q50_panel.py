"""Does ANY quantity separate the good Gln50 variants from the bad ones?

The complementarity metric failed this test (p = 0.571, worse group ranked
higher). Before concluding that catalytic quality is invisible to these models, it
is worth asking whether the information is present in the structures at all.

Kries et al. attribute the activity difference to active sites that are tightly
packed (Gln, Met, Phe) versus water-filled (His, Ser, Ala, Lys). So the panel
includes a purely crystallographic measure — how many ordered waters sit in the
oxyanion pocket — as a positive control on the LABELS. If waters separate the
groups and no model quantity does, the information is in the structure and the
models are not using it. If nothing separates them, the labels are not recoverable
from these crystals at all.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

PACKED = {"Q50(wt)": "4BS0", "Q50M": "6TU6", "Q50F": "6SRW"}
WATERY = {"Q50H": "6SRZ", "Q50S": "6SRY", "Q50A": "6SS1", "Q50K": "6SS3"}
RES50 = 50  # the varied position, author numbering


def waters_near_ligand(pdb, cache, comp="6NT", cutoff=5.0):
    """Ordered waters within `cutoff` of the bound ligand copy in chain A."""
    path = os.path.join(cache, f"{pdb}.pdb")
    lig, wat = [], []
    for ln in open(path):
        if ln.startswith("HETATM"):
            name = ln[17:20].strip()
            xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
            if name == comp and ln[21] == "A":
                lig.append(xyz)
            elif name == "HOH":
                wat.append(xyz)
    if not lig or not wat:
        return np.nan
    lig, wat = np.array(lig), np.array(wat)
    d = np.linalg.norm(wat[:, None, :] - lig[None, :, :], axis=-1).min(1)
    return float((d < cutoff).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reorg_perresidue.json")
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--site", type=float, default=6.0)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import MPNNWrapper, ProteinMPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER, get_ligand, seq_struct_from_ligand
    from preorganization import clean_pdb
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    params = os.path.join(a.src, "bayes_design", "ligand_mpnn", "model_params")
    ckpt = os.path.join(params, "ligandmpnn_v_32_010_25.pt")
    with open(a.json) as f:
        store = json.load(f)
    mpnn = ProteinMPNNWrapper(device=device)

    rows = []
    for group, members in (("packed", PACKED), ("watery", WATERY)):
        for v, pdb in members.items():
            stem, _, _ = clean_pdb(pdb, "6NT", a.cache)
            ctx = get_ligand(stem, structures_dir=a.cache, chain_list=("A",))
            seq, struct = seq_struct_from_ligand(ctx)
            y = np.asarray(ctx["Y"])
            holo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                               checkpoint_path=ckpt, context=ctx, use_ligand=True)
            apo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                              checkpoint_path=ckpt, context=ctx, use_ligand=False)
            p_h = per_position_probs(holo, seq, struct, True, 8)
            p_a = per_position_probs(apo, seq, struct, True, 8)
            p_m = per_position_probs(mpnn, seq, struct, True, 8)
            del holo, apo
            torch.cuda.empty_cache()

            bb = np.asarray(struct)
            d = np.linalg.norm(bb[:, :, None, :] - y[None, None, :, :], axis=-1).min((1, 2))
            site = d < a.site
            nat = np.array([aa20.index(c) for c in seq])
            idx = np.arange(len(seq))
            j = int(np.argmin(d))  # residue closest to the ligand, as a proxy for pos 50
            rows.append({
                "variant": v, "group": group, "pdb": pdb,
                "complementarity": float((p_h * (np.log(p_h + 1e-12)
                                                 - np.log(p_a + 1e-12))).sum(1)[site].mean()),
                "logp_holo_site": float(np.log(p_h[idx, nat] + 1e-12)[site].mean()),
                "logp_apo_site": float(np.log(p_a[idx, nat] + 1e-12)[site].mean()),
                "logp_mpnn_site": float(np.log(p_m[idx, nat] + 1e-12)[site].mean()),
                "entropy_holo": float((-(p_h * np.log(p_h + 1e-12)).sum(1))[site].mean()),
                "logp_closest": float(np.log(p_h[j, nat[j]] + 1e-12)),
                "n_site": int(site.sum()),
                "waters_in_pocket": waters_near_ligand(pdb, a.cache),
            })
            print(f"  {v:<9} {group:<7} done", flush=True)

    keys = ["complementarity", "logp_holo_site", "logp_apo_site", "logp_mpnn_site",
            "entropy_holo", "logp_closest", "waters_in_pocket"]
    print("\n" + "=" * 76)
    print("Which quantity separates good (Gln/Met/Phe) from bad (His/Ser/Ala/Lys)?")
    print("=" * 76)
    print(f"{'quantity':>18} {'packed mean':>12} {'watery mean':>12} "
          f"{'two-sided p':>12}  ranking (best->worst)")
    for k in keys:
        p = [r[k] for r in rows if r["group"] == "packed" and np.isfinite(r[k])]
        w = [r[k] for r in rows if r["group"] == "watery" and np.isfinite(r[k])]
        if len(p) < 2 or len(w) < 2:
            print(f"{k:>18}   insufficient data")
            continue
        u = stats.mannwhitneyu(p, w)
        order = sorted([r for r in rows if np.isfinite(r[k])], key=lambda r: -r[k])
        tag = "".join("P" if r["group"] == "packed" else "w" for r in order)
        print(f"{k:>18} {np.mean(p):>12.4f} {np.mean(w):>12.4f} {u.pvalue:>12.3f}  {tag}")
    print("\n  ranking string: P = packed (better catalyst), w = water-filled (worse).")
    print("  perfect separation would read PPPwwww or wwwwPPP.")
    print("  best attainable two-sided p with 3 vs 4 is 0.057.")
    json.dump(rows, open("q50_panel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
