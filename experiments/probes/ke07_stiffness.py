"""Did directed evolution rigidify the Kemp eliminase active site?

KE07 was evolved over 7 rounds with crystal structures at rounds 0, 5, 6 and 7
(Tawfik lab). If the extractable catalytic signal in these models is *geometric
precision* — how stiffly the sequence holds the catalytic constellation — then the
active-site stiffness should rise along the trajectory.

Measured per structure, restricted to the residues contacting the bound ligand in
5D2T (A9, I11, S48, F49, W50, E101, Y128, H201, K222 — the known KE07 site, with
E101 the catalytic base):

  mpnn_msf   mean-square fluctuation at the site from the ProteinMPNN Hessian
             (sequence-aware; lower = stiffer)
  anm_msf    same from an elastic network (structure-only control: if this tracks
             equally well, the signal is backbone geometry, not sequence)
  geo_spec   log p(native) at the site minus its value under jittered backbones —
             how much the site's residue preferences depend on exact geometry
  logp_site  plain log p(native) at the site

6C7H is the deliberately-inactivated W50A design mutant: a negative control that
should look worse, not better.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy import stats

BB = ["N", "CA", "C", "O"]
SITE = [9, 11, 48, 49, 50, 101, 128, 201, 222]
CORE = [50, 101, 222]  # Trp50, catalytic base Glu101, Lys222
N_RIGID = 6


def extract(path):
    model = next(iter(PDBParser(QUIET=True).get_structure("x", path)))
    for chain in model:
        seq, xyz, nums = [], [], []
        for res in chain:
            if res.id[0] != " " or res.get_resname() not in protein_letters_3to1:
                continue
            if not all(x in res for x in BB):
                continue
            seq.append(protein_letters_3to1[res.get_resname()])
            xyz.append([res[x].get_coord() for x in BB])
            nums.append(res.id[1])
        if len(seq) > 100:
            return "".join(seq), np.array(xyz, np.float32), np.array(nums)
    return None


def anm_hessian(coords, cutoff=13.0):
    ca = coords[:, 1].astype(np.float64)
    ell = len(ca)
    h = np.zeros((3 * ell, 3 * ell))
    for i in range(ell):
        for j in range(i + 1, ell):
            r = ca[j] - ca[i]
            d = np.linalg.norm(r)
            if d > cutoff:
                continue
            blk = np.outer(r, r) / d**2
            h[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -blk
            h[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = -blk
            h[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += blk
            h[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += blk
    return h


def msf_at(stiff, ell, idx):
    """Mean-square fluctuation at selected residues, from a stiffness matrix."""
    w, u = np.linalg.eigh(stiff)
    keep = (np.arange(len(w)) >= N_RIGID) & (w > 1e-8)
    if keep.sum() < 10:
        return np.nan
    msf = ((u[:, keep] ** 2) / w[keep]).sum(1).reshape(ell, 3).sum(1)
    return float(np.mean(msf[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_ke07")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import ProteinMPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    wrapper = ProteinMPNNWrapper(device=device)
    model = wrapper.model
    rng = np.random.default_rng(0)
    os.makedirs(a.cache, exist_ok=True)

    with open("ke07.json") as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        path = os.path.join(a.cache, f"{e['pdb']}.pdb")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{e['pdb']}.pdb", path)
        got = extract(path)
        if got is None:
            continue
        seq, coords, nums = got
        if any(c not in aa20 for c in seq):
            continue
        pos = {n: i for i, n in enumerate(nums)}
        idx = [pos[n] for n in SITE if n in pos]
        core = [pos[n] for n in CORE if n in pos]
        if len(idx) < len(SITE) - 1:
            print(f"skip {e['pdb']}: site incomplete ({len(idx)}/{len(SITE)})")
            continue
        ell = len(seq)
        seq_idx = torch.tensor([[AMINO_ACID_ORDER.index(c) for c in seq]], device=device)

        # Full reduced Hessian of log p(seq|struct) via Hessian-vector products.
        x = torch.tensor(coords, device=device).unsqueeze(0).requires_grad_(True)
        _, logits = model(
            X=x, S=seq_idx, mask=torch.ones(1, ell, device=device),
            chain_M=torch.ones(1, ell, device=device),
            residue_idx=torch.arange(ell, device=device).unsqueeze(0),
            chain_encoding_all=torch.ones(1, ell, device=device),
            use_input_decoding_order=True, randn=None,
            decoding_order=torch.arange(ell, device=device).unsqueeze(0),
        )
        lp = torch.log_softmax(logits, -1).gather(2, seq_idx.unsqueeze(-1)).sum()
        (g,) = torch.autograd.grad(lp, x, create_graph=True)
        h = np.zeros((3 * ell, 3 * ell))
        for i in range(ell):
            for ax in range(3):
                v = torch.zeros_like(x)
                v[0, i, :, ax] = 1.0
                (hv,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
                h[3 * i + ax] = hv[0].sum(1).reshape(-1).detach().cpu().numpy()
        h = 0.5 * (h + h.T)

        mpnn_msf = msf_at(-h, ell, idx)
        anm_msf = msf_at(anm_hessian(coords), ell, idx)

        # Sequence-only measures at the site.
        p0 = per_position_probs(wrapper, seq, torch.tensor(coords), True, 32)
        nat = np.array([aa20.index(c) for c in seq])
        lp_site = float(np.mean(np.log(p0[idx, nat[idx]] + 1e-12)))
        lp_core = float(np.mean(np.log(p0[core, nat[core]] + 1e-12))) if core else np.nan
        blur = []
        for _ in range(a.samples):
            noisy = coords + rng.normal(0, a.sigma, coords.shape).astype(np.float32)
            pk = per_position_probs(wrapper, seq, torch.tensor(noisy), True, 32)
            blur.append(np.mean(np.log(pk[idx, nat[idx]] + 1e-12)))
        geo_spec = lp_site - float(np.mean(blur))

        rows.append({**e, "L": ell, "mpnn_msf": mpnn_msf, "anm_msf": anm_msf,
                     "geo_spec": geo_spec, "logp_site": lp_site, "logp_core": lp_core})
        print(f"{e['pdb']} r{e['round']} L={ell} mpnn_msf={mpnn_msf:.4g} "
              f"anm_msf={anm_msf:.4g} geo={geo_spec:+.3f} logp={lp_site:+.3f}", flush=True)

    ctrl = [r for r in rows if r["pdb"] == "6C7H"]
    traj = [r for r in rows if r["pdb"] != "6C7H" and r["round"] is not None]
    print(f"\n{len(traj)} structures across rounds "
          f"{sorted({r['round'] for r in traj})}\n")
    print(f"{'metric':>12} {'r0':>10} {'r5':>10} {'r6':>10} {'r7':>10}   "
          f"{'Spearman(round, metric)':>26}")
    for key in ("mpnn_msf", "anm_msf", "geo_spec", "logp_site", "logp_core"):
        by = {}
        for r in traj:
            by.setdefault(r["round"], []).append(r[key])
        means = [np.nanmean(by.get(k, [np.nan])) for k in (0, 5, 6, 7)]
        v = np.array([r[key] for r in traj], float)
        rd = np.array([r["round"] for r in traj], float)
        ok = np.isfinite(v)
        rho = stats.spearmanr(rd[ok], v[ok])
        print(f"{key:>12} " + " ".join(f"{m:>10.4g}" for m in means) +
              f"   rho={rho.statistic:+.3f}  p={rho.pvalue:.3g}")
    if ctrl:
        c = ctrl[0]
        print(f"\nnegative control 6C7H (W50A, inactivated): mpnn_msf={c['mpnn_msf']:.4g} "
              f"geo={c['geo_spec']:+.3f} logp_site={c['logp_site']:+.3f}")
    json.dump(rows, open("ke07_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
