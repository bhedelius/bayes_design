"""How many independent axes does the term library actually have?

`CombinedModel` searches a weighted log-linear space over its terms. That space is
only as big as the terms are independent - and several of them are trained on the
same data for the same task, so they may be near-duplicates. This scores every
available backend on the same positions and reports pairwise agreement plus the
participation ratio of the term correlation matrix (the effective number of
independent terms).

If the effective dimensionality is much smaller than the term count, then most of
the objective space is degenerate and a weight search would mostly be exploring
directions that do not change the design.
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures400.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--params", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=160)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import MPNNWrapper, ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    params = a.params or os.path.join(a.src, "bayes_design", "ligand_mpnn", "model_params")

    with open(a.inp) as f:
        records = [r for r in json.load(f)
                   if all(c in aa20 for c in r["seq"]) and len(r["seq"]) <= a.max_len][: a.n]

    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)
    # Context-bound backends are built per protein (they bind their structure at
    # construction), so hold the checkpoints and rebuild inside the loop.
    ckpt = {"protein_mpnn": os.path.join(params, "proteinmpnn_v_48_020.pt"),
            "soluble_mpnn": os.path.join(params, "solublempnn_v_48_020.pt")}

    names = ["ProteinMPNN_v030", "ProteinMPNN_v020", "SolubleMPNN", "ProtXLNet", "background"]
    cols = {n: [] for n in names}

    counts = np.zeros(20)
    for rec in records:
        for c in rec["seq"]:
            counts[aa20.index(c)] += 1
    bg = counts / counts.sum()

    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"], dtype=np.float32)
        struct = torch.tensor(coords)
        ctx = {
            "X": struct.unsqueeze(0) if False else struct,
            "S": torch.tensor([aa20.index(c) for c in seq]),
            "mask": torch.ones(len(seq)),
            "chain_mask": torch.ones(len(seq)),
            "R_idx": torch.arange(len(seq)),
            "chain_labels": torch.ones(len(seq)),
            "Y": torch.zeros(1, 3), "Y_t": torch.zeros(1), "Y_m": torch.zeros(1),
        }
        try:
            p = {
                "ProteinMPNN_v030": per_position_probs(mpnn, seq, struct, True, 32),
                "ProteinMPNN_v020": per_position_probs(
                    MPNNWrapper(device=device, model_type="protein_mpnn",
                                checkpoint_path=ckpt["protein_mpnn"], context=ctx),
                    seq, struct, True, 32),
                "SolubleMPNN": per_position_probs(
                    MPNNWrapper(device=device, model_type="soluble_mpnn",
                                checkpoint_path=ckpt["soluble_mpnn"], context=ctx),
                    seq, struct, True, 32),
                "ProtXLNet": per_position_probs(xlnet, seq, struct, False, 32),
                "background": np.repeat(bg[None], len(seq), 0),
            }
        except Exception as e:  # noqa: BLE001
            print(f"skip {rec['pdb']}: {type(e).__name__}: {e}", flush=True)
            continue
        for n in names:
            cols[n].append(np.log(p[n] + 1e-9))
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={len(seq)}", flush=True)

    mats = {n: np.concatenate(cols[n]) for n in names}
    live = [n for n in names if len(mats[n])]
    # Centre each position's log-probs so only the *shape* of the preference counts.
    z = {n: (mats[n] - mats[n].mean(1, keepdims=True)).reshape(-1) for n in live}

    print(f"\nPairwise correlation of centred log-probabilities ({len(mats[live[0]])} positions)")
    c = np.zeros((len(live), len(live)))
    for i, n1 in enumerate(live):
        for j, n2 in enumerate(live):
            c[i, j] = 1.0 if i == j else np.corrcoef(z[n1], z[n2])[0, 1]
    print("     " + " ".join(f"{n[:9]:>10}" for n in live))
    for i, n1 in enumerate(live):
        print(f"{n1[:16]:>16} " + " ".join(f"{c[i, j]:>10.3f}" for j in range(len(live))))

    print("\nTop-1 agreement (how often two terms pick the same residue)")
    for i, n1 in enumerate(live):
        for j, n2 in enumerate(live):
            if j <= i:
                continue
            print(f"  {n1:>17} vs {n2:<17} {(mats[n1].argmax(1) == mats[n2].argmax(1)).mean() * 100:5.1f}%")

    w = np.linalg.eigvalsh(c)[::-1]
    pr = w.sum() ** 2 / (w**2).sum()
    print(f"\nEigenvalues of the term correlation matrix: {np.round(w, 3)}")
    print(f"Participation ratio (effective independent terms): {pr:.2f} out of {len(live)}")
    print(f"Variance in the top 2 directions: {w[:2].sum() / w.sum() * 100:.1f}%")


if __name__ == "__main__":
    main()
