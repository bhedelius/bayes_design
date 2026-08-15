"""Where do catalytic residues sit in the (structure, sequence) plane?

Hypothesis: a catalytic residue is present for function, not for the fold, so the
structure term should *dislike* it while sequence conservation *likes* it — i.e.
strongly negative PMI. If that holds, PMI is a per-residue selector: optimize the
structure-determined positions, protect the function-determined ones.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

BB = ["N", "CA", "C", "O"]


def virtual_cb(coords):
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def extract(path, chain_id):
    """Sequence, backbone coords and author residue numbers for one chain."""
    model = next(iter(PDBParser(QUIET=True).get_structure("x", path)))
    if chain_id not in model:
        return None
    seq, xyz, nums = [], [], []
    for res in model[chain_id]:
        if res.id[0] != " " or res.get_resname() not in protein_letters_3to1:
            continue
        if not all(x in res for x in BB):
            continue
        seq.append(protein_letters_3to1[res.get_resname()])
        xyz.append([res[x].get_coord() for x in BB])
        nums.append(res.id[1])
    return "".join(seq), np.array(xyz, float), np.array(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="catalytic.json")
    ap.add_argument("--out", default="catalytic_scored.csv")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_enz")
    ap.add_argument("--balance", type=float, default=0.002)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    os.makedirs(a.cache, exist_ok=True)
    device = torch.device(a.device)
    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)

    with open(a.inp) as f:
        entries = json.load(f)

    rows = []
    for k, e in enumerate(entries):
        path = os.path.join(a.cache, f"{e['pdb']}.pdb")
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{e['pdb']}.pdb", path)
            got = extract(path, e["chain"])
        except Exception as ex:  # noqa: BLE001
            print(f"skip {e['pdb']}: {type(ex).__name__}")
            continue
        if got is None:
            continue
        seq, coords, nums = got
        if not (60 <= len(seq) <= 400) or any(c not in aa20 for c in seq):
            continue
        cat_idx = [int(np.where(nums == rn)[0][0]) for rn in e["catalytic_resnum"] if (nums == rn).any()]
        if len(cat_idx) != len(e["catalytic_resnum"]):
            print(f"skip {e['pdb']}: catalytic residue not resolved in chain")
            continue

        struct = torch.tensor(coords, dtype=torch.float32)
        p_s = per_position_probs(mpnn, seq, struct, True, 32)
        p_q = per_position_probs(xlnet, seq, struct, False, 32)
        nat = np.array([aa20.index(c) for c in seq])
        idx = np.arange(len(seq))
        ls = np.log(p_s[idx, nat] + 1e-12)
        lq = np.log(p_q[idx, nat] + 1e-12)
        cb = virtual_cb(coords)
        nbrs = (np.linalg.norm(cb[:, None] - cb[None], axis=-1) < 10.0).sum(1) - 1

        for i in range(len(seq)):
            rows.append({"pdb": e["pdb"], "pos": i, "aa": seq[i], "nbrs": float(nbrs[i]),
                         "logp_struct": ls[i], "logp_seq": lq[i], "pmi": ls[i] - lq[i],
                         "catalytic": int(i in cat_idx)})
        print(f"[{k + 1}/{len(entries)}] {e['pdb']} L={len(seq)} catalytic={len(cat_idx)}", flush=True)

    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} residues -> {a.out}")


if __name__ == "__main__":
    main()
