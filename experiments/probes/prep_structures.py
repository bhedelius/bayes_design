"""Download PDB entries and extract per-residue backbone coordinates + B-factors."""

import argparse
import json
import os
import urllib.request

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

BB = ["N", "CA", "C", "O"]


def fetch(pdb_id, cache):
    path = os.path.join(cache, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
    return path


def extract(path, min_len):
    """Return (seq, coords L x 4 x 3, bfac L, chain) for the first usable protein chain."""
    model = next(iter(PDBParser(QUIET=True).get_structure("x", path)))
    for chain in model:
        seq, xyz, bfac = [], [], []
        for res in chain:
            if res.id[0] != " ":  # hetero / water
                continue
            name = res.get_resname()
            if name not in protein_letters_3to1 or not all(a in res for a in BB):
                continue
            atoms = [res[a] for a in BB]
            seq.append(protein_letters_3to1[name])
            xyz.append([a.get_coord() for a in atoms])
            bfac.append(float(np.mean([a.get_bfactor() for a in atoms])))
        if len(seq) >= min_len:
            return "".join(seq), np.array(xyz, dtype=float), np.array(bfac), chain.id
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="ids.txt")
    ap.add_argument("--out", default="structures.json")
    ap.add_argument("--cache", default="pdbs")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    with open(a.ids) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    records, dropped = [], {}
    for pdb_id in ids:
        try:
            out = extract(fetch(pdb_id, a.cache), a.min_len)
        except Exception as e:  # noqa: BLE001
            dropped[pdb_id] = f"fetch/parse ({type(e).__name__})"
            continue
        if out is None:
            dropped[pdb_id] = "no usable chain"
            continue
        seq, xyz, bfac, chain = out
        if len(seq) > a.max_len:
            dropped[pdb_id] = f"too long ({len(seq)})"
            continue
        if bfac.std() < 1e-6:
            dropped[pdb_id] = "flat B-factors"
            continue
        records.append({"pdb": pdb_id, "chain": chain, "seq": seq,
                        "coords": xyz.tolist(), "bfac": bfac.tolist()})

    with open(a.out, "w") as f:
        json.dump(records, f)
    print(f"kept {len(records)}/{len(ids)} structures -> {a.out}")
    if dropped:
        reasons = {}
        for v in dropped.values():
            reasons[v.split(" (")[0]] = reasons.get(v.split(" (")[0], 0) + 1
        print("dropped:", reasons)


if __name__ == "__main__":
    main()
