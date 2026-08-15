"""H1, done properly: does the active site move less on binding as evolution proceeds?

Preorganization is a statement about *coordinates*: a preorganized site is already
in its bound conformation before the ligand arrives, so binding costs no
reorganization. My earlier test masked the ligand while keeping the holo backbone,
which never varied the geometry — so it measured ligand complementarity, not
preorganization.

KE07 has apo and ligand-bound crystals of the same variant at rounds 0, 5 and 7,
which is exactly the comparison needed.

  structural  : active-site RMSD between apo and holo after global superposition
                (no model involved - the direct measurement)
  model       : KL between p(a | X_apo) and p(a | X_holo) at the site, ligand
                absent in both, so only backbone geometry differs

Prediction (Warshel, and the crystallographers' own reading of this system):
both DECREASE along the trajectory.
"""

import argparse
import itertools
import json
import os
import sys
import urllib.request

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy import stats

BB = ["N", "CA", "C", "O"]
APO = {0: ["4Z08", "5D2V"], 5: ["5D2X", "5D2Y", "5D30", "6C7M", "6C7T", "6C7V"],
       7: ["5D33", "5D37", "5D38"]}
HOLO = {0: ["5D2T", "5D2W"], 5: ["6DKV"], 7: ["6DC1", "6CT3"]}
LIG = "6VP"


def load(pdb, cache):
    path = os.path.join(cache, f"{pdb}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", path)
    model = next(iter(PDBParser(QUIET=True).get_structure("x", path)))
    ch = model["A"] if "A" in model else next(iter(model))
    seq, xyz, nums = [], [], []
    for res in ch:
        if res.id[0] != " " or res.get_resname() not in protein_letters_3to1:
            continue
        if not all(x in res for x in BB):
            continue
        seq.append(protein_letters_3to1[res.get_resname()])
        xyz.append([res[x].get_coord() for x in BB])
        nums.append(res.id[1])
    lig = np.array([a.get_coord() for r in ch if r.get_resname() == LIG for a in r])
    return "".join(seq), np.array(xyz, np.float32), np.array(nums), lig


def kabsch(mob, ref):
    """Rotation+translation putting `mob` onto `ref` (both N x 3)."""
    mc, rc = mob.mean(0), ref.mean(0)
    h = (mob - mc).T @ (ref - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, rc - r @ mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_ke07")
    ap.add_argument("--site", type=float, default=6.0)
    ap.add_argument("--model", action="store_true", help="also run the model comparison")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    if a.model:
        sys.path.insert(0, a.src)
        sys.path.insert(0, ".")
        import torch

        from bayes_design.model import ProteinMPNNWrapper
        from bayes_design.utils import AMINO_ACID_ORDER
        from score_conditionals import per_position_probs

        aa20 = AMINO_ACID_ORDER[:20]
        mpnn = ProteinMPNNWrapper(device=torch.device(a.device))

    cache_p = {}

    def get(pdb):
        if pdb not in cache_p:
            cache_p[pdb] = load(pdb, a.cache)
        return cache_p[pdb]

    rows = []
    for rnd in (0, 5, 7):
        for ap_pdb, ho_pdb in itertools.product(APO[rnd], HOLO[rnd]):
            try:
                sa, xa, na, _ = get(ap_pdb)
                sh, xh, nh, lig = get(ho_pdb)
                if len(lig) == 0:
                    continue
                common = np.intersect1d(na, nh)
                ia = np.array([int(np.where(na == n)[0][0]) for n in common])
                ih = np.array([int(np.where(nh == n)[0][0]) for n in common])
                # Superimpose on all shared CA, then measure only the site.
                r, t = kabsch(xa[ia, 1], xh[ih, 1])
                xa_al = (xa[:, :, :] @ r.T) + t
                d_lig = np.linalg.norm(xh[ih][:, :, None, :] - lig[None, None, :, :],
                                       axis=-1).min((1, 2))
                site = d_lig < a.site
                if site.sum() < 4:
                    continue
                dev = np.linalg.norm(xa_al[ia][site, 1] - xh[ih][site, 1], axis=-1)
                allv = np.linalg.norm(xa_al[ia][:, 1] - xh[ih][:, 1], axis=-1)
                row = {"round": rnd, "apo": ap_pdb, "holo": ho_pdb,
                       "n_site": int(site.sum()), "n_common": len(common),
                       "site_rmsd": float(np.sqrt((dev**2).mean())),
                       "all_rmsd": float(np.sqrt((allv**2).mean()))}

                if a.model:
                    pa = per_position_probs(mpnn, sa, torch.tensor(xa), True, 16)
                    ph = per_position_probs(mpnn, sh, torch.tensor(xh), True, 16)
                    pa_c, ph_c = pa[ia], ph[ih]
                    kl = (ph_c * (np.log(ph_c + 1e-12) - np.log(pa_c + 1e-12))).sum(1)
                    row["model_kl_site"] = float(kl[site].mean())
                    row["model_kl_rest"] = float(kl[~site].mean())
                rows.append(row)
                print(f"r{rnd}  {ap_pdb}->{ho_pdb}  site={row['n_site']:>2}  "
                      f"site_RMSD={row['site_rmsd']:.3f}  all_RMSD={row['all_rmsd']:.3f}"
                      + (f"  model_KL={row['model_kl_site']:.4f}" if a.model else ""),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"r{rnd} {ap_pdb}->{ho_pdb}: FAILED {type(e).__name__}: {e}")

    json.dump(rows, open("apo_holo_results.json", "w"), indent=1)
    print("\n" + "=" * 66)
    print("Does the active site move LESS on binding as evolution proceeds?")
    print("=" * 66)
    keys = ["site_rmsd", "all_rmsd"] + (["model_kl_site", "model_kl_rest"] if a.model else [])
    rd = np.array([r["round"] for r in rows], float)
    print(f"{'metric':>15} " + "".join(f"{'r' + str(k):>10}" for k in (0, 5, 7)) +
          f"{'Spearman':>11}{'p':>9}   n_pairs")
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        means = [v[rd == q].mean() for q in (0, 5, 7)]
        rho = stats.spearmanr(rd, v)
        print(f"{k:>15} " + "".join(f"{m:>10.3f}" for m in means) +
              f"{rho.statistic:>+11.3f}{rho.pvalue:>9.3g}   {len(v)}")
    print("\n  site_rmsd is the direct structural measure of preorganization:")
    print("  lower = the apo site already sits in its bound conformation.")


if __name__ == "__main__":
    main()
