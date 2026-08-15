"""Average over a structural ensemble instead of dividing by a sequence model.

Caliby (Shuai et al. 2025) removes non-structural signal by conditioning on an
ensemble of backbones and averaging the per-structure energies:

    E_ens(S) = (1/K) sum_k E_k(S)   <=>   p_ens(S) ∝ prod_k p_k(S)^(1/K)

That is a product-of-experts, i.e. exactly a `CombinedModel` with K positive terms
of weight 1/K — no new machinery, just a noised-context backend. They report the
same trick works with plain ProteinMPNN ("tied sampling") and that cheap Gaussian
noise is a usable ensemble generator, with an optimum near ~1 A ensemble RMSD.

This compares, on identical positions:
  - ProteinMPNN on the crystal backbone
  - ProteinMPNN averaged over a Gaussian ensemble (the Caliby-style objective)
  - ProteinMPNN divided by the blurred ensemble (what we tried before — the wrong sign)
  - BayesDesign

The signature to look for is the one Caliby reports: native recovery DOWN while the
structural signal (burial) is PRESERVED. BayesDesign drops recovery too, but takes
burial down with it.
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
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--samples", type=int, default=16)  # K = 16, as in Caliby
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

    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    out = {}

    def blank():
        return {"rec": [], "wcm": [], "burial": [], "H": [], "pssm": [], "rmsd": []}

    for s in sigmas:
        out[f"ens s={s}"] = blank()
        out[f"div s={s}"] = blank()
    out["ProteinMPNN"] = blank()
    out["BayesDesign"] = blank()

    def record(key, P, nat, nbrs, prof, rmsd=np.nan):
        out[key]["rec"].append((P.argmax(1) == nat).mean())
        out[key]["wcm"].append(wcm[P.argmax(1)].mean())
        out[key]["burial"].append(stats.spearmanr(nbrs, P @ kdv).statistic)
        out[key]["H"].append((-(P * np.log(P + 1e-12)).sum(1)).mean())
        # Jensen-Shannon between the designed profile and the native profile, as a
        # stand-in for Caliby's PSSM-similarity check: are we still making
        # statistically sensible proteins even as recovery falls?
        q = P.mean(0)
        m = 0.5 * (prof + q)
        js = 0.5 * (prof * np.log((prof + 1e-12) / (m + 1e-12))).sum() + \
             0.5 * (q * np.log((q + 1e-12) / (m + 1e-12))).sum()
        out[key]["pssm"].append(1 - js / np.log(2))
        out[key]["rmsd"].append(rmsd)

    for k, rec in enumerate(records):
        seq, coords = rec["seq"], np.array(rec["coords"], dtype=np.float32)
        struct = torch.tensor(coords)
        nat = np.array([aa20.index(c) for c in seq])
        prof = np.bincount(nat, minlength=20) / len(nat)
        nbrs = neighbor_count(virtual_cb(coords))

        p_s = per_position_probs(mpnn, seq, struct, True, 32)
        p_q = per_position_probs(xlnet, seq, struct, False, 32)
        record("ProteinMPNN", p_s, nat, nbrs, prof)
        r = (p_s + a.balance) / (p_q + a.balance)
        record("BayesDesign", r / r.sum(1, keepdims=True), nat, nbrs, prof)

        for s in sigmas:
            if s == 0.0:
                record(f"ens s={s}", p_s, nat, nbrs, prof, 0.0)
                record(f"div s={s}", p_s, nat, nbrs, prof, 0.0)
                continue
            logsum, blur, disp = np.zeros_like(p_s), np.zeros_like(p_s), []
            for _ in range(a.samples):
                noisy = coords + rng.normal(0, s, coords.shape).astype(np.float32)
                # Ensemble RMSD after removing the rigid-body component (CA only).
                d = noisy[:, 1] - coords[:, 1]
                disp.append(np.sqrt((np.linalg.norm(d - d.mean(0), axis=1) ** 2).mean()))
                pk = per_position_probs(mpnn, seq, torch.tensor(noisy), True, 32)
                logsum += np.log(pk + 1e-12)
                blur += pk
            # Product of experts: geometric mean over the ensemble.
            ens = np.exp(logsum / a.samples)
            ens /= ens.sum(1, keepdims=True)
            record(f"ens s={s}", ens, nat, nbrs, prof, float(np.mean(disp)))
            blur /= a.samples
            g = (p_s + a.balance) / (blur + a.balance)
            record(f"div s={s}", g / g.sum(1, keepdims=True), nat, nbrs, prof, float(np.mean(disp)))
        print(f"[{k + 1}/{len(records)}] {rec['pdb']} L={len(seq)}", flush=True)

    print(f"\nK={a.samples} Gaussian-noised backbones, {len(records)} proteins\n")
    print(f"{'objective':>14} {'ens RMSD':>9} {'recovery':>9} {'W/C/M':>7} {'burial':>8} "
          f"{'entropy':>8} {'profile sim':>12}")
    for key in ["ProteinMPNN", "BayesDesign"] + [f"ens s={s}" for s in sigmas] + \
               [f"div s={s}" for s in sigmas]:
        v = out[key]
        if not v["rec"]:
            continue
        print(f"{key:>14} {np.nanmean(v['rmsd']):>9.2f} {np.mean(v['rec']) * 100:>8.1f}% "
              f"{np.mean(v['wcm']) * 100:>6.1f}% {np.mean(v['burial']):>+8.3f} "
              f"{np.mean(v['H']):>8.3f} {np.mean(v['pssm']):>12.3f}")
    print("\n'ens' = average log-probs over the ensemble (Caliby-style product of experts)")
    print("'div' = divide by the ensemble-averaged distribution (what we tried before)")


if __name__ == "__main__":
    main()
