"""A denominator made of blurred geometry instead of sequence.

    log p(a | X) - log p(a | X + noise)

Both terms are the same inverse-folding model, so this asks: how much of this
residue's preference depends on getting the backbone *exactly* right, rather than
on the fold being roughly this shape? Residues whose preference survives 1 A of
jitter are generic to the fold; residues that need precise geometry are the
conformationally specific ones.

Motivation: the sequence-referenced denominator cancels burial (both terms can see
it) and imports a language model's memorisation of the native sequence. This
version never touches a sequence model. Whether it is a *new* axis or collapses
onto the existing structure axis is the thing to measure.
"""

import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="structures400.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--sigma", type=float, default=1.0, help="backbone jitter, Angstrom")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--balance", type=float, default=0.002)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch
    from scipy import stats

    from bayes_design.model import ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from score_conditionals import neighbor_count, per_position_probs, virtual_cb

    aa20 = AMINO_ACID_ORDER[:20]
    KD = dict(zip("ACDEFGHIKLMNPQRSTVWY",
                  [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
    kdv = np.array([KD[c] for c in aa20])
    wcm = np.array([c in "WCM" for c in aa20])
    device = torch.device(a.device)
    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)
    rng = np.random.default_rng(0)

    with open(a.inp) as f:
        records = [r for r in json.load(f)
                   if all(c in aa20 for c in r["seq"]) and len(r["seq"]) <= a.max_len][: a.n]

    agg = {k: [] for k in ("geo", "bayes", "mpnn", "seq")}
    burial = {k: [] for k in agg}
    rec_rate = {k: [] for k in agg}
    wcm_rate = {k: [] for k in agg}
    corr = []

    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"], dtype=np.float32)
        struct = torch.tensor(coords)
        p_s = per_position_probs(mpnn, seq, struct, True, 32)
        p_q = per_position_probs(xlnet, seq, struct, False, 32)

        # Denominator: the model's expectation under blurred geometry.
        blur = np.zeros_like(p_s)
        for _ in range(a.samples):
            noisy = coords + rng.normal(0, a.sigma, coords.shape).astype(np.float32)
            blur += per_position_probs(mpnn, seq, torch.tensor(noisy), True, 32)
        blur /= a.samples

        dists = {}
        dists["mpnn"] = p_s
        dists["seq"] = p_q
        r = (p_s + a.balance) / (p_q + a.balance)
        dists["bayes"] = r / r.sum(1, keepdims=True)
        g = (p_s + a.balance) / (blur + a.balance)
        dists["geo"] = g / g.sum(1, keepdims=True)

        nat = np.array([aa20.index(c) for c in seq])
        nbrs = neighbor_count(virtual_cb(coords))
        for key, P in dists.items():
            agg[key].append(np.log(P + 1e-12) - np.log(P + 1e-12).mean(1, keepdims=True))
            burial[key].append(stats.spearmanr(nbrs, P @ kdv).statistic)
            rec_rate[key].append((P.argmax(1) == nat).mean())
            wcm_rate[key].append(wcm[P.argmax(1)].mean())
        corr.append(np.corrcoef(
            (np.log(dists["geo"] + 1e-12) - np.log(dists["geo"] + 1e-12).mean(1, keepdims=True)).ravel(),
            (np.log(dists["bayes"] + 1e-12) - np.log(dists["bayes"] + 1e-12).mean(1, keepdims=True)).ravel(),
        )[0, 1])
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={len(seq)}", flush=True)

    print(f"\nGeometric-specificity denominator: blurred backbone, sigma={a.sigma} A, "
          f"{a.samples} samples, {len(records)} proteins\n")
    print(f"{'objective':>12} {'recovery':>10} {'W/C/M':>8} {'burial signal':>15}")
    for key, lab in (("mpnn", "ProteinMPNN"), ("seq", "ProtXLNet"),
                     ("bayes", "BayesDesign"), ("geo", "geometric")):
        print(f"{lab:>12} {np.mean(rec_rate[key]) * 100:>9.1f}% {np.mean(wcm_rate[key]) * 100:>7.1f}% "
              f"{np.mean(burial[key]):>+15.3f}")

    print("\nIs it a new axis? correlation of centred log-probabilities")
    mats = {k: np.concatenate(v).ravel() for k, v in agg.items()}
    keys = ["mpnn", "seq", "bayes", "geo"]
    print("        " + " ".join(f"{k:>9}" for k in keys))
    c = np.zeros((4, 4))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            c[i, j] = 1.0 if i == j else np.corrcoef(mats[k1], mats[k2])[0, 1]
        print(f"{k1:>8} " + " ".join(f"{c[i, j]:>9.3f}" for j in range(4)))
    w = np.linalg.eigvalsh(c)[::-1]
    print(f"\nParticipation ratio with the geometric term: {w.sum() ** 2 / (w ** 2).sum():.2f} of 4")
    c3 = c[:3, :3]
    w3 = np.linalg.eigvalsh(c3)[::-1]
    print(f"                          without it: {w3.sum() ** 2 / (w3 ** 2).sum():.2f} of 3")


if __name__ == "__main__":
    main()
