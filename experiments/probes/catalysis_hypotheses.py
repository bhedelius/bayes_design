"""Test three theory-derived hypotheses about what makes a good catalyst.

Each is evaluated on matched pairs: the same designed enzyme before and after
directed evolution, crystallised with the SAME ligand, so the only variable is the
sequence that evolution selected.

H1  PREORGANIZATION (Warshel).  Catalysis is the reorganization energy you avoid.
    A preorganized site does not need to change between the unliganded and the
    ligand-bound state, so its residue preferences should barely move.
        reorg = KL[ p(a | struct, ligand) || p(a | struct, ligand masked) ]
    Prediction: the better catalyst has the LOWER reorganization.

H2  CONDITIONAL FRUSTRATION (Wolynes / Ferreiro).  A catalytic residue is a
    residue the fold does not want but the substrate makes welcome.
        relief = log p(native | ligand) - log p(native | ligand masked)
    Prediction: large and positive at catalytic residues, and LARGER in the better
    catalyst. Must also lose the signal in a known dead mutant.

H3  LANDSCAPE FLATNESS (Albery & Knowles).  A perfect catalyst has no deep well
    along the cycle; product inhibition is a deep well.
        flatness = -variance over states of log p(seq | state)
    Prediction: the better catalyst has LOWER variance across apo / substrate /
    product structures.

The same two distributions feed H1 and H2 through different functionals — one is
the shape of the preference, the other the native residue's likelihood — so they
can move independently. That is the informative part.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

# Matched pairs: (label, worse variant, better variant, ligand state, fold improvement)
PAIRS = [
    ("HG3 -> HG3.17", "3NYD", "4BS0", "TSA", 1800),
    ("KE59 R1 -> R13", "3UXD", "3UZ5", "TSA", 2000),
    ("KE07 r0 -> r5", "5D2W", "6DKV", "product", 200),
]
# Single-position series on the HG3.17 background, all TSA-bound.
Q50 = {"6SRW": "Q50F", "6SRY": "Q50S", "6SRZ": "Q50H",
       "6SS1": "Q50A", "6SS3": "Q50K", "6TU6": "Q50M", "4BS0": "Q50(wt)"}
DEAD = {"6C7H": "KE07 W50A (inactivated)"}


def fetch(pdb, cache):
    path = os.path.join(cache, f"{pdb}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--params", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--site", type=float, default=6.0, help="A from ligand = active site")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="catalysis_results.json")
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

    def score(pdb):
        """Return per-position conditionals with the ligand seen and masked."""
        fetch(pdb, a.cache)
        ctx = get_ligand(pdb, structures_dir=a.cache, chain_list=("A",))
        seq, struct = seq_struct_from_ligand(ctx)
        if any(c not in aa20 for c in seq):
            raise ValueError("non-canonical residue")
        y = ctx["Y"].numpy() if hasattr(ctx["Y"], "numpy") else np.asarray(ctx["Y"])
        ym = ctx["Y_m"].numpy() if hasattr(ctx["Y_m"], "numpy") else np.asarray(ctx["Y_m"])
        y = y[ym.astype(bool)] if ym.size == len(y) else y
        if len(y) == 0:
            raise ValueError("no ligand atoms parsed")
        holo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                           checkpoint_path=ckpt, context=ctx, use_ligand=True)
        apo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                          checkpoint_path=ckpt, context=ctx, use_ligand=False)
        p_h = per_position_probs(holo, seq, struct, True, a.batch)
        p_a = per_position_probs(apo, seq, struct, True, a.batch)
        del holo, apo
        torch.cuda.empty_cache()
        # Active site = residues with ANY backbone atom within `site` A of the ligand
        # (CA-only was far too strict: it found 1 residue where all-atom finds 16).
        bb = struct.numpy() if hasattr(struct, "numpy") else np.asarray(struct)
        d = np.linalg.norm(bb[:, :, None, :] - y[None, None, :, :], axis=-1).min((1, 2))
        site = d < a.site
        return seq, p_h, p_a, site, d

    def measures(seq, p_h, p_a, site):
        nat = np.array([aa20.index(c) for c in seq])
        idx = np.arange(len(seq))
        lp_h = np.log(p_h[idx, nat] + 1e-12)
        lp_a = np.log(p_a[idx, nat] + 1e-12)
        kl = (p_h * (np.log(p_h + 1e-12) - np.log(p_a + 1e-12))).sum(1)
        return {
            "n_site": int(site.sum()),
            "H1_reorg_site": float(kl[site].mean()),
            "H1_reorg_rest": float(kl[~site].mean()),
            "H2_relief_site": float((lp_h - lp_a)[site].mean()),
            "H2_relief_rest": float((lp_h - lp_a)[~site].mean()),
            "logp_holo_site": float(lp_h[site].mean()),
            "logp_apo_site": float(lp_a[site].mean()),
        }

    results = {"pairs": [], "q50": [], "dead": []}

    print("=" * 74)
    print("H1 / H2 on matched design -> evolved pairs (same ligand, same fold)")
    print("=" * 74)
    for label, worse, better, lig, fold in PAIRS:
        row = {"label": label, "ligand": lig, "fold": fold}
        ok = True
        for tag, pdb in (("worse", worse), ("better", better)):
            try:
                m = measures(*score(pdb)[:4])
            except Exception as e:  # noqa: BLE001
                print(f"  {label}: {pdb} failed ({type(e).__name__}: {e})")
                ok = False
                break
            row[tag] = {"pdb": pdb, **m}
        if not ok:
            continue
        results["pairs"].append(row)
        w, b = row["worse"], row["better"]
        print(f"\n{label}  ({fold}x better, {lig}-bound, site={w['n_site']}/{b['n_site']} res)")
        print(f"  H1 reorganization at site  {w['pdb']}={w['H1_reorg_site']:.4f}  "
              f"{b['pdb']}={b['H1_reorg_site']:.4f}   "
              f"{'PREDICTED (lower)' if b['H1_reorg_site'] < w['H1_reorg_site'] else 'against'}")
        print(f"  H2 frustration relief      {w['pdb']}={w['H2_relief_site']:+.4f}  "
              f"{b['pdb']}={b['H2_relief_site']:+.4f}   "
              f"{'PREDICTED (higher)' if b['H2_relief_site'] > w['H2_relief_site'] else 'against'}")
        print(f"     (rest of chain: reorg {w['H1_reorg_rest']:.4f}/{b['H1_reorg_rest']:.4f}, "
              f"relief {w['H2_relief_rest']:+.4f}/{b['H2_relief_rest']:+.4f})")

    print("\n" + "=" * 74)
    print("Q50 series on the HG3.17 background (all TSA-bound, single position)")
    print("=" * 74)
    for pdb, name in Q50.items():
        try:
            m = measures(*score(pdb)[:4])
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} {name}: failed ({type(e).__name__})")
            continue
        results["q50"].append({"pdb": pdb, "variant": name, **m})
        print(f"  {pdb} {name:<9} reorg={m['H1_reorg_site']:.4f}  "
              f"relief={m['H2_relief_site']:+.4f}  logp_holo={m['logp_holo_site']:+.4f}")

    print("\n" + "=" * 74)
    print("Negative control: a deliberately inactivated variant")
    print("=" * 74)
    for pdb, name in DEAD.items():
        try:
            m = measures(*score(pdb)[:4])
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} {name}: failed ({type(e).__name__}: {e})")
            continue
        results["dead"].append({"pdb": pdb, "variant": name, **m})
        print(f"  {pdb} {name}: reorg={m['H1_reorg_site']:.4f} relief={m['H2_relief_site']:+.4f}")

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
