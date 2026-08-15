"""Is BayesDesign's effect local?

Redesign a contiguous patch with BayesDesign, splice the designed residues into
the native sequence, then re-score the *untouched* positions with ProteinMPNN.
If the effect is local, ProteinMPNN's preferences outside the patch are unchanged.
If rigidity propagates, they shift toward the patch's character.

Reported as a function of 3D distance from the patch, so propagation through space
(not just along the chain) is visible.
"""

import argparse
import json
import sys

import numpy as np

FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346,
    "Q": 0.493, "E": 0.497, "G": 0.544, "H": 0.323, "I": 0.462,
    "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509,
    "S": 0.507, "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}
PATCH = 20


def virtual_cb(coords):
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures400.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--balance", type=float, default=0.002)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    import torch

    from bayes_design.model import ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    flex_v = np.array([FLEX[c] for c in aa20])
    device = torch.device(a.device)
    sys.path.insert(0, ".")
    from score_conditionals import per_position_probs

    with open(a.inp) as f:
        records = [r for r in json.load(f) if all(c in aa20 for c in r["seq"])][: a.n]

    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)

    # Distance shells (Angstrom) from the patch, measured CB-to-CB.
    edges = [0, 6, 8, 10, 12, 15, 20, 1e9]
    changed = [[] for _ in edges[:-1]]
    dflex = [[] for _ in edges[:-1]]
    inpatch = []

    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"])
        ell = len(seq)
        if ell < PATCH + 40:
            continue
        struct = torch.tensor(coords, dtype=torch.float32)

        p_s = per_position_probs(mpnn, seq, struct, True, 32)
        p_q = per_position_probs(xlnet, seq, struct, False, 32)
        bayes = (p_s + a.balance) / (p_q + a.balance)
        bayes /= bayes.sum(1, keepdims=True)

        # Patch centred on the most buried stretch, so it is a real core patch.
        cb = virtual_cb(coords)
        dist = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
        nbrs = (dist < 10.0).sum(1)
        win = np.convolve(nbrs, np.ones(PATCH) / PATCH, mode="valid")
        start = int(win.argmax())
        sl = slice(start, start + PATCH)

        # Splice BayesDesign's picks into the patch; native everywhere else.
        designed = list(seq)
        for i in range(start, start + PATCH):
            designed[i] = aa20[bayes[i].argmax()]
        designed = "".join(designed)

        # Re-score the untouched positions with ProteinMPNN in the new context.
        p_s2 = per_position_probs(mpnn, designed, struct, True, 32)

        outside = np.ones(ell, bool)
        outside[sl] = False
        d_to_patch = dist[:, sl].min(1)
        before, after = p_s[outside].argmax(1), p_s2[outside].argmax(1)
        dd = d_to_patch[outside]
        for j in range(len(edges) - 1):
            m = (dd >= edges[j]) & (dd < edges[j + 1])
            if m.sum():
                changed[j].append((before[m] != after[m]).mean())
                dflex[j].append(flex_v[after[m]].mean() - flex_v[before[m]].mean())
        # Direct effect inside the patch, for scale.
        inpatch.append(flex_v[bayes[sl].argmax(1)].mean() - flex_v[p_s[sl].argmax(1)].mean())
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} patch {start}-{start + PATCH}", flush=True)

    print("\nEffect on UNTOUCHED positions, by 3D distance from the redesigned patch")
    print(f"{'shell (A)':>12} {'n prot':>7} {'MPNN pick changed':>18} {'delta flexibility':>18}")
    for j in range(len(edges) - 1):
        if not changed[j]:
            continue
        lab = f"{edges[j]}-{edges[j + 1]}" if edges[j + 1] < 1e8 else f">{edges[j]}"
        print(f"{lab:>12} {len(changed[j]):>7} {np.mean(changed[j]) * 100:>17.1f}% "
              f"{np.mean(dflex[j]):>+13.4f} +/- {np.std(dflex[j],ddof=1)/np.sqrt(len(dflex[j])):.4f}")
    print(f"\nDirect effect INSIDE the patch (BayesDesign vs ProteinMPNN pick): "
          f"{np.mean(inpatch):+.4f} +/- {np.std(inpatch,ddof=1)/np.sqrt(len(inpatch)):.4f}")


if __name__ == "__main__":
    main()
