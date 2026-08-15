"""Score every residue of each structure with the BayesDesign component models and
write a per-residue table alongside the crystallographic B-factor.

For each position i we take the one-out conditional (`bidirectional_mlm` masking):
    structure term  p(a_i | backbone, seq_{-i})   -- ProteinMPNN
    sequence term   p(a_i | seq_{-i})             -- ProtXLNet marginal
    bayes term      p_struct / p_seq, renormalized -- the BayesDesign objective
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

# Bhaskaran & Ponnuswamy average flexibility index (the ProtParam scale), keyed by
# the ACDEFGHIKLMNPQRSTVWY ordering used by AMINO_ACID_ORDER.
FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346,
    "Q": 0.493, "E": 0.497, "G": 0.544, "H": 0.323, "I": 0.462,
    "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509,
    "S": 0.507, "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}


def virtual_cb(coords):
    """Ideal CB position from N, CA, C (defined for Gly too); a burial proxy."""
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def neighbor_count(cb, cutoff=10.0):
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    return ((d < cutoff).sum(1) - 1).astype(float)


def per_position_probs(model, seq, struct, is_struct_model, batch):
    """Return an L x 20 array of one-out conditional probabilities."""
    ell = len(seq)
    order = list(range(ell))
    out = np.zeros((ell, 20))
    for start in range(0, ell, batch):
        toks = list(range(start, min(start + batch, ell)))
        kwargs = dict(seq=[seq] * len(toks), decode_order=order,
                      token_to_decode=toks, mask_type="bidirectional_mlm")
        if is_struct_model:
            kwargs["struct"] = struct.clone()
        out[start : start + len(toks)] = model(**kwargs).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures.json")
    ap.add_argument("--out", default="residues.csv")
    ap.add_argument("--src", default=os.environ.get("BAYES_SRC", ""))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    # Matches CombinedModel's default; damps the ratio where p_seq is tiny.
    ap.add_argument("--balance", type=float, default=0.002)
    # Dump the raw L x 20 conditionals so downstream questions (constrained
    # decoding, alternative denominators) need no model re-run.
    ap.add_argument("--dump-probs", default="")
    a = ap.parse_args()

    if a.src:
        sys.path.insert(0, a.src)
    from bayes_design.model import ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)

    with open(a.inp) as f:
        records = json.load(f)
    records = records[a.shard :: a.nshards]

    # Background amino-acid frequency over the whole (unsharded) set, for the null
    # denominator. Computed from every record so shards agree on it.
    with open(a.inp) as f:
        counts = np.zeros(20)
        for rec in json.load(f):
            for c in rec["seq"]:
                if c in aa20:
                    counts[aa20.index(c)] += 1
    bg = (counts / counts.sum())[None, :]

    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)

    rows, dump_struct, dump_seq = [], [], []
    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"])
        if any(c not in aa20 for c in seq):
            print(f"skip {rec['pdb']}: non-canonical residue", flush=True)
            continue
        struct = torch.tensor(coords, dtype=torch.float32)
        t0 = time.time()
        p_struct = per_position_probs(mpnn, seq, struct, True, a.batch)
        p_seq = per_position_probs(xlnet, seq, struct, False, a.batch)

        native = np.array([aa20.index(c) for c in seq])
        idx = np.arange(len(seq))
        bayes = (p_struct + a.balance) / (p_seq + a.balance)
        bayes = bayes / bayes.sum(1, keepdims=True)
        raw = p_struct / p_seq
        raw = raw / raw.sum(1, keepdims=True)
        # Null denominator: the same ratio against a position-independent background
        # frequency table. If this reproduces BayesDesign's picks, the contextual
        # sequence model is contributing nothing beyond frequency correction.
        freqc = (p_struct + a.balance) / (bg + a.balance)
        freqc = freqc / freqc.sum(1, keepdims=True)

        def logp(p):
            return np.log(p[idx, native] + 1e-12)

        def ent(p):
            return -(p * np.log(p + 1e-12)).sum(1)

        nc = neighbor_count(virtual_cb(coords))
        gly, pro = aa20.index("G"), aa20.index("P")
        # Expected flexibility of the designed residue under the full distribution,
        # rather than of the single argmax pick.
        flex_vec = np.array([FLEX[c] for c in aa20])
        eflex_struct = p_struct @ flex_vec
        eflex_seq = p_seq @ flex_vec
        eflex_bayes = bayes @ flex_vec
        for i in range(len(seq)):
            rows.append({
                "pdb": rec["pdb"], "pos": i, "aa": seq[i],
                "bfac": rec["bfac"][i], "nbrs": nc[i],
                "logp_struct": logp(p_struct)[i], "logp_seq": logp(p_seq)[i],
                "logp_bayes": logp(bayes)[i], "logp_rawratio": logp(raw)[i],
                "H_struct": ent(p_struct)[i], "H_seq": ent(p_seq)[i], "H_bayes": ent(bayes)[i],
                "p_gly_struct": p_struct[i, gly], "p_gly_bayes": bayes[i, gly],
                "p_pro_struct": p_struct[i, pro], "p_pro_bayes": bayes[i, pro],
                "eflex_struct": eflex_struct[i], "eflex_seq": eflex_seq[i],
                "eflex_bayes": eflex_bayes[i],
                "logp_freqcorr": logp(freqc)[i], "H_freqcorr": ent(freqc)[i],
                "argmax_struct": aa20[p_struct[i].argmax()],
                "argmax_bayes": aa20[bayes[i].argmax()],
                "argmax_freqcorr": aa20[freqc[i].argmax()],
                "argmax_seq": aa20[p_seq[i].argmax()],
            })
        if a.dump_probs:
            dump_struct.append(p_struct.astype(np.float32))
            dump_seq.append(p_seq.astype(np.float32))
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={len(seq)} {time.time() - t0:.1f}s", flush=True)

    if a.dump_probs:
        np.savez_compressed(a.dump_probs, p_struct=np.concatenate(dump_struct),
                            p_seq=np.concatenate(dump_seq), bg=bg[0])
        print(f"dumped conditionals -> {a.dump_probs}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} residues -> {a.out}")


if __name__ == "__main__":
    main()
