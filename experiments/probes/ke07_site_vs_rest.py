"""Is the drop in log p specific to the active site, or protein-wide?

The trajectory run found that ProteinMPNN likes the KE07 active site less with each
round of directed evolution. That is only interesting if it is *site-specific*.

KE07 was designed with Rosetta, which — like ProteinMPNN — favours idealised
packing, so the starting design sits at a model-favourable point and any mutation
moves away from it. If evolution lowered log p everywhere equally, the site result
is trivial. If the drop is concentrated at the catalytic residues, evolution was
specifically paying likelihood at the active site to buy catalysis.

Cheap: no Hessian, just one scoring pass per structure.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

SITE = [9, 11, 48, 49, 50, 101, 128, 201, 222]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="pdbs_ke07")
    ap.add_argument("--shell", type=float, default=10.0, help="A around the site")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import ProteinMPNNWrapper
    from bayes_design.utils import AMINO_ACID_ORDER
    from ke07_stiffness import extract
    from score_conditionals import per_position_probs, virtual_cb

    aa20 = AMINO_ACID_ORDER[:20]
    device = torch.device(a.device)
    mpnn = ProteinMPNNWrapper(device=device)

    with open("ke07.json") as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        path = os.path.join(a.cache, f"{e['pdb']}.pdb")
        if not os.path.exists(path):
            continue
        got = extract(path)
        if got is None:
            continue
        seq, coords, nums = got
        if any(c not in aa20 for c in seq):
            continue
        pos = {n: i for i, n in enumerate(nums)}
        idx = np.array([pos[n] for n in SITE if n in pos])
        if len(idx) < len(SITE) - 1:
            continue

        p = per_position_probs(mpnn, seq, torch.tensor(coords), True, 32)
        nat = np.array([aa20.index(c) for c in seq])
        lp = np.log(p[np.arange(len(seq)), nat] + 1e-12)

        cb = virtual_cb(coords)
        d = np.linalg.norm(cb[:, None] - cb[None], axis=-1)[:, idx].min(1)
        site = np.zeros(len(seq), bool)
        site[idx] = True
        shell = (~site) & (d < a.shell)
        far = d >= a.shell

        rows.append({**e, "site": lp[site].mean(), "shell": lp[shell].mean(),
                     "far": lp[far].mean(), "all": lp.mean(),
                     "n_shell": int(shell.sum()), "n_far": int(far.sum())})
        print(f"{e['pdb']} r{e['round']} site={rows[-1]['site']:+.3f} "
              f"shell={rows[-1]['shell']:+.3f} far={rows[-1]['far']:+.3f}", flush=True)

    ctrl = [r for r in rows if r["pdb"] == "6C7H"]
    traj = [r for r in rows if r["pdb"] != "6C7H" and r["round"] is not None]
    rd = np.array([r["round"] for r in traj], float)

    print(f"\n{len(traj)} structures. Mean log p(native) by region and round:\n")
    print(f"{'region':>10} {'r0':>9} {'r5':>9} {'r6':>9} {'r7':>9}  "
          f"{'Spearman(round)':>16} {'p':>9}")
    for key, lab in (("site", "active site"), ("shell", f"<{a.shell:.0f}A shell"),
                     ("far", f">{a.shell:.0f}A rest"), ("all", "whole chain")):
        by = {}
        for r in traj:
            by.setdefault(r["round"], []).append(r[key])
        means = [np.mean(by.get(k, [np.nan])) for k in (0, 5, 6, 7)]
        v = np.array([r[key] for r in traj])
        rho = stats.spearmanr(rd, v)
        print(f"{lab:>10} " + " ".join(f"{m:>9.3f}" for m in means) +
              f"  {rho.statistic:>+16.3f} {rho.pvalue:>9.3g}")

    # The decisive contrast: site relative to the rest of the same structure.
    rel = np.array([r["site"] - r["far"] for r in traj])
    rho = stats.spearmanr(rd, rel)
    print(f"\n  site MINUS far, by round: " +
          " ".join(f"r{k}={np.mean([r['site'] - r['far'] for r in traj if r['round'] == k]):+.3f}"
                   for k in (0, 5, 6, 7)))
    print(f"  Spearman(round, site-far) = {rho.statistic:+.3f}  p={rho.pvalue:.3g}")
    if ctrl:
        c = ctrl[0]
        print(f"\n  negative control 6C7H (W50A): site={c['site']:+.3f} far={c['far']:+.3f} "
              f"site-far={c['site'] - c['far']:+.3f}")


if __name__ == "__main__":
    main()
