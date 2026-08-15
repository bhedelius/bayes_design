"""How far from an active site must a redesign stay before it stops perturbing it?

For each enzyme, redesign a fixed number of positions drawn from a single distance
shell around the annotated catalytic residues, splice them in, then re-score the
*untouched* catalytic residues with ProteinMPNN. The result is a perturbation-vs-
distance curve: the safe radius for stabilising mutations.

Shell size is held constant (same number of mutations everywhere) so outer shells
are not penalised for simply containing more residues.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

EDGES = [(0, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 20), (20, 1e9)]
K = 8  # mutations per shell, fixed across shells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="catalytic.json")
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_enz")
    ap.add_argument("--balance", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import ProteinMPNNWrapper, XLNetWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from catalytic_pmi import extract, virtual_cb
    from score_conditionals import per_position_probs

    aa20 = AMINO_ACID_ORDER[:20]
    os.makedirs(a.cache, exist_ok=True)
    device = torch.device(a.device)
    mpnn = ProteinMPNNWrapper(device=device)
    xlnet = XLNetWrapper(device=device)
    rng = np.random.default_rng(a.seed)

    with open(a.inp) as f:
        entries = json.load(f)

    # results[objective][shell] = list of per-enzyme perturbations
    dlogp = {o: [[] for _ in EDGES] for o in ("bayes", "mpnn")}
    flipped = {o: [[] for _ in EDGES] for o in ("bayes", "mpnn")}
    nshell = [[] for _ in EDGES]

    for k, e in enumerate(entries):
        path = os.path.join(a.cache, f"{e['pdb']}.pdb")
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{e['pdb']}.pdb", path)
            got = extract(path, e["chain"])
        except Exception:  # noqa: BLE001
            continue
        if got is None:
            continue
        seq, coords, nums = got
        if not (60 <= len(seq) <= 400) or any(c not in aa20 for c in seq):
            continue
        cat = [int(np.where(nums == rn)[0][0]) for rn in e["catalytic_resnum"] if (nums == rn).any()]
        if len(cat) != len(e["catalytic_resnum"]) or not cat:
            continue

        struct = torch.tensor(coords, dtype=torch.float32)
        p_s = per_position_probs(mpnn, seq, struct, True, 32)
        p_q = per_position_probs(xlnet, seq, struct, False, 32)
        bayes = (p_s + a.balance) / (p_q + a.balance)
        bayes /= bayes.sum(1, keepdims=True)

        nat = np.array([aa20.index(c) for c in seq])
        cb = virtual_cb(coords)
        dist = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
        d_cat = dist[:, cat].min(1)
        base_lp = np.log(p_s[cat, nat[cat]] + 1e-12)
        base_pick = p_s[cat].argmax(1)

        for j, (lo, hi) in enumerate(EDGES):
            pool = np.where((d_cat >= lo) & (d_cat < hi))[0]
            pool = np.array([i for i in pool if i not in cat])
            if len(pool) < K:
                continue
            sel = rng.choice(pool, K, replace=False)
            nshell[j].append(len(pool))
            for obj, P in (("bayes", bayes), ("mpnn", p_s)):
                designed = list(seq)
                for i in sel:
                    designed[i] = aa20[P[i].argmax()]
                p_s2 = per_position_probs(mpnn, "".join(designed), struct, True, 32)
                dlogp[obj][j].append(float(np.mean(np.log(p_s2[cat, nat[cat]] + 1e-12) - base_lp)))
                flipped[obj][j].append(float(np.mean(p_s2[cat].argmax(1) != base_pick)))
        print(f"[{k + 1}/{len(entries)}] {e['pdb']} L={len(seq)} cat={len(cat)}", flush=True)

    def summ(v):
        return f"{np.mean(v):+.3f} +/- {np.std(v, ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else "  n/a"

    print(f"\nEffect on the UNTOUCHED catalytic residues of mutating {K} positions in one shell")
    print(f"{'shell (A)':>11} {'n enz':>6} {'pool':>6} | {'BayesDesign dlogP':>22} {'flip':>7} |"
          f" {'ProteinMPNN dlogP':>22} {'flip':>7}")
    for j, (lo, hi) in enumerate(EDGES):
        if not dlogp["bayes"][j]:
            continue
        lab = f"{lo}-{hi}" if hi < 1e8 else f">{lo}"
        print(f"{lab:>11} {len(dlogp['bayes'][j]):>6} {np.mean(nshell[j]):>6.0f} |"
              f" {summ(dlogp['bayes'][j]):>22} {np.mean(flipped['bayes'][j]) * 100:>6.1f}% |"
              f" {summ(dlogp['mpnn'][j]):>22} {np.mean(flipped['mpnn'][j]) * 100:>6.1f}%")


if __name__ == "__main__":
    main()
