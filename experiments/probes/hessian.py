"""The full Hessian of log p(seq | struct) in backbone space, via autograd.

The diagonal second-difference probe (`curvature.py`) only sees local stiffness.
The off-diagonal terms are the couplings, and together they define normal modes:

    treat  -log p  as an energy  =>  -H is the stiffness matrix
    mean-square fluctuation_i    =  sum_k  |u_k[i]|^2 / lambda_k   over non-rigid modes

That is the standard elastic-network relation, so the prediction is directly
comparable to an ANM baseline built from contacts alone. The question this
answers: has ProteinMPNN learned mechanics beyond what the contact graph gives
you for free?

Hessian is taken in a reduced basis - one coordinate per residue per axis, moving
that residue's four backbone atoms together - so it costs 3L Hessian-vector
products rather than a full (12L)^2 finite-difference sweep.
"""

import argparse
import json
import sys

import numpy as np
import torch
from scipy import stats

N_RIGID = 6  # translations + rotations


def reduced_hessian(model, coords, seq_idx, device):
    """3L x 3L Hessian of log p(seq|struct); residue i moved rigidly along each axis."""
    ell = coords.shape[0]
    x = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    mask = torch.ones(1, ell, device=device)
    order = torch.arange(ell, device=device).unsqueeze(0)
    _, logits = model(
        X=x, S=seq_idx, mask=mask,
        chain_M=torch.ones(1, ell, device=device),
        residue_idx=torch.arange(ell, device=device).unsqueeze(0),
        chain_encoding_all=torch.ones(1, ell, device=device),
        use_input_decoding_order=True, randn=None, decoding_order=order,
    )
    lp = torch.log_softmax(logits, -1).gather(2, seq_idx.unsqueeze(-1)).sum()
    (g,) = torch.autograd.grad(lp, x, create_graph=True)

    h = np.zeros((3 * ell, 3 * ell), dtype=np.float64)
    for i in range(ell):
        for ax in range(3):
            v = torch.zeros_like(x)
            v[0, i, :, ax] = 1.0  # move all four backbone atoms of residue i together
            (hv,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True)
            h[3 * i + ax] = hv[0].sum(1).reshape(-1).detach().cpu().numpy()
    return 0.5 * (h + h.T)  # symmetrize away autograd noise


def msf_from_hessian(stiff, ell):
    """Mean-square fluctuation per residue from a stiffness matrix."""
    w, u = np.linalg.eigh(stiff)
    keep = np.arange(len(w)) >= N_RIGID
    keep &= w > 1e-8  # drop rigid modes and any non-positive (unstable) directions
    if keep.sum() < 10:
        return None, 0.0
    msf = ((u[:, keep] ** 2) / w[keep]).sum(1).reshape(ell, 3).sum(1)
    return msf, float((w[N_RIGID:] > 1e-8).mean())


def anm(coords, cutoff=13.0):
    """Anisotropic network model on CA atoms - the contacts-only baseline."""
    ca = coords[:, 1]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures400.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--ckpt", default="")  # vanilla ProteinMPNN weights; default is v_48_030
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from bayes_design.model import ProteinMPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    w = ProteinMPNNWrapper(device=device)
    if a.ckpt:
        # Same architecture, different training-noise level: v_48_002 saw 0.02 A of
        # backbone jitter, v_48_030 saw 0.30 A. Coordinate sensitivity should differ.
        ck = torch.load(a.ckpt, map_location=device)
        w.model.load_state_dict(ck["model_state_dict"])
        print(f"loaded {a.ckpt} (training noise {ck['noise_level']} A)")
    model = w.model

    with open(a.inp) as f:
        records = [r for r in json.load(f)
                   if all(c in aa20 for c in r["seq"]) and len(r["seq"]) <= a.max_len][: a.n]

    res = {"mpnn": [], "anm": [], "diag": [], "combo": []}
    stable = []
    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"], dtype=np.float32)
        ell = len(seq)
        seq_idx = torch.tensor([[AMINO_ACID_ORDER.index(c) for c in seq]], device=device)
        h = reduced_hessian(model, coords, seq_idx, device)

        logb = np.log(np.array(rec["bfac"]))
        msf, frac = msf_from_hessian(-h, ell)
        if msf is None:
            print(f"[{k + 1}] {rec['pdb']} unstable, skipped", flush=True)
            continue
        stable.append(frac)
        msf_anm, _ = msf_from_hessian(anm(coords), ell)
        # Diagonal-only stiffness, for comparison with the finite-difference probe.
        diag = -np.array([h[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].trace() for i in range(ell)])
        with np.errstate(divide="ignore"):
            inv_diag = np.where(diag > 0, 1.0 / np.maximum(diag, 1e-12), np.nan)

        r_m = stats.spearmanr(logb, msf).statistic
        r_a = stats.spearmanr(logb, msf_anm).statistic
        ok = np.isfinite(inv_diag)
        r_d = stats.spearmanr(logb[ok], inv_diag[ok]).statistic
        r_c = stats.spearmanr(logb, stats.rankdata(msf) + stats.rankdata(msf_anm)).statistic
        for key, val in (("mpnn", r_m), ("anm", r_a), ("diag", r_d), ("combo", r_c)):
            res[key].append(val)
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={ell} pos-modes={frac:.2f} "
              f"MPNN={r_m:+.3f} ANM={r_a:+.3f} diag={r_d:+.3f}", flush=True)

    print(f"\nPer-protein Spearman(log B, predicted fluctuation), n={len(res['mpnn'])}")
    for key, lab in (("diag", "ProteinMPNN Hessian, diagonal only"),
                     ("mpnn", "ProteinMPNN Hessian, full normal modes"),
                     ("anm", "ANM baseline (contacts only)"),
                     ("combo", "ANM + ProteinMPNN (rank sum)")):
        v = np.array(res[key])
        v = v[np.isfinite(v)]
        print(f"  {lab:40s} {v.mean():+.3f} +/- {v.std(ddof=1) / np.sqrt(len(v)):.3f}")
    d = np.array(res["mpnn"]) - np.array(res["anm"])
    d = d[np.isfinite(d)]
    print(f"\n  MPNN - ANM: {d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}  "
          f"p={stats.wilcoxon(d).pvalue:.2g}")
    print(f"  mean fraction of non-rigid modes that are stable: {np.mean(stable):.3f}")


if __name__ == "__main__":
    main()
