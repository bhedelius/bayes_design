"""Is log p(seq | struct) energy-like? Measure its curvature in backbone space.

If log p behaved like -G/kT, then in a harmonic approximation its Hessian with
respect to backbone coordinates would set the fluctuation amplitudes, and
crystallographic B-factors would follow 1 / stiffness.

Per residue we displace its backbone atoms by +/- eps along several random
directions and take the second difference of the whole-sequence log-likelihood:

    curvature_i = < logP(+eps) + logP(-eps) - 2 logP(0) > / eps^2

Negative curvature means the native geometry sits at a local maximum (stiff);
the predicted mean-square fluctuation is 1 / (-curvature).

This is a second-order probe. Correlating log p(native residue) with B-factor -
the zeroth-order version - mostly recovers burial; this asks whether the model's
log-probability surface has energy-like geometry at all.
"""

import argparse
import json
import sys

import numpy as np
import torch
from scipy import stats

DIRS = 3  # random displacement directions per residue


def logp_batch(model, struct_batch, seq_idx, device):
    """Whole-sequence log-likelihood for a batch of perturbed backbones.

    One forward pass per batch: ProteinMPNN returns logits for every position
    under the given decoding order, so summing log p(native_i) over positions
    gives log p(seq | struct) for each structure in the batch.
    """
    b, ell = struct_batch.shape[0], struct_batch.shape[1]
    s = seq_idx.expand(b, ell).to(device)
    mask = torch.isfinite(struct_batch.sum((2, 3))).float().to(device)
    x = torch.nan_to_num(struct_batch).to(device)
    order = torch.arange(ell, device=device).expand(b, ell)
    with torch.no_grad():
        _, logits = model(
            X=x, S=s, mask=mask,
            chain_M=torch.ones(b, ell, device=device),
            residue_idx=torch.arange(ell, device=device).expand(b, ell),
            chain_encoding_all=torch.ones(b, ell, device=device),
            use_input_decoding_order=True, randn=None, decoding_order=order,
        )
        lp = torch.log_softmax(logits, -1)
    return lp.gather(2, s.unsqueeze(-1)).squeeze(-1).sum(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures400.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.3)  # Angstrom, ~thermal scale
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="curvature.csv")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from bayes_design.model import ProteinMPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    model = ProteinMPNNWrapper(device=device).model
    rng = np.random.default_rng(0)

    with open(a.inp) as f:
        records = [r for r in json.load(f) if all(c in aa20 for c in r["seq"])][: a.n]

    rows, per_prot = [], []
    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"], dtype=np.float32)
        ell = len(seq)
        seq_idx = torch.tensor([[AMINO_ACID_ORDER.index(c) for c in seq]], device=device)
        base_t = torch.tensor(coords).unsqueeze(0)
        base = logp_batch(model, base_t, seq_idx, device)[0]

        # Build every perturbed structure: residue i, direction d, sign +/-.
        dirs = rng.normal(size=(ell, DIRS, 3))
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        jobs = [(i, d, s) for i in range(ell) for d in range(DIRS) for s in (1, -1)]
        vals = np.zeros(len(jobs))
        for start in range(0, len(jobs), a.batch):
            chunk = jobs[start : start + a.batch]
            batch = np.repeat(coords[None], len(chunk), 0)
            for m, (i, d, s) in enumerate(chunk):
                batch[m, i] += (s * a.eps * dirs[i, d]).astype(np.float32)
            vals[start : start + len(chunk)] = logp_batch(
                model, torch.tensor(batch), seq_idx, device)

        v = vals.reshape(ell, DIRS, 2)
        curv = ((v[:, :, 0] + v[:, :, 1] - 2 * base) / a.eps**2).mean(1)
        rows += [{"pdb": rec["pdb"], "pos": i, "aa": seq[i], "bfac": rec["bfac"][i],
                  "curv": curv[i]} for i in range(ell)]
        stiff = -curv
        ok = stiff > 0
        if ok.sum() > 10:
            r = stats.spearmanr(np.log(rec["bfac"])[ok] if isinstance(rec["bfac"], np.ndarray)
                                else np.log(np.array(rec["bfac"]))[ok], (1.0 / stiff[ok]))
            per_prot.append(r.statistic)
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={ell} "
              f"stiff frac={ok.mean():.2f} rho={per_prot[-1] if per_prot else float('nan'):+.3f}", flush=True)

    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    d = np.array(per_prot)
    print(f"\nIs the native backbone a local maximum of log p? "
          f"fraction of residues with negative curvature: "
          f"{np.mean([r['curv'] < 0 for r in rows]):.3f}")
    print(f"Per-protein Spearman(log B, predicted fluctuation 1/stiffness): "
          f"{d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}  (n={len(d)})")
    print(f"wrote {len(rows)} residues -> {a.out}")


if __name__ == "__main__":
    main()
