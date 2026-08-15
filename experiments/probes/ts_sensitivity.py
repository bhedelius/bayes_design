"""How far must a ligand move before LigandMPNN notices?

The differential objective for catalysis - score a sequence with the ligand in
transition-state geometry, minus the same sequence with it in ground-state
geometry - targets the quantity that actually sets rate, ddG* = dG(TS) - dG(GS),
rather than the single-state quantity every inverse-folding model reports. But it
is only computable if the model resolves the two geometries at all.

LigandMPNN sees a ligand as coordinates plus element indices with no bond graph
(`Y`, `Y_t`, `Y_m`). So a GS->TS change reduces *entirely* to a coordinate
displacement, and the feasibility question becomes quantitative and answerable
before any docking or pose building: how much of the model's total ligand signal
does a displacement of d Angstrom recover?

Three reference scales, all measured here rather than assumed:

  CEILING   Removing the ligand outright (use_ligand=False) is the largest change
            available, so it defines 100% of the ligand signal.
  FLOOR     The same perturbation read at positions >15 A from the ligand, where
            it should do nothing. Anything at or below this is leakage.
  DATA      The ligand's own crystallographic positional uncertainty, from its
            atomic B-factors: u_rms = sqrt(B / (8 pi^2)). A GS->TS displacement
            smaller than this is below the resolution of the input coordinates,
            independent of the model - no scoring function can recover it.

Read the curve against the chemistry. A proton transfer with partial bond
elongation moves heavy atoms ~0.2-0.5 A; a planar-to-tetrahedral change at a
carbonyl moves them ~0.5-0.8 A. Whichever displacement the differential must
resolve has to clear both the FLOOR and the DATA scale to be worth building
poses for.
"""

import argparse
import json
import os
import sys

import numpy as np


def kl_rows(p, q):
    """Per-position KL(p||q) for two (n, 20) probability arrays."""
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


def ligand_bfactors(pdb_path, comp):
    """RMS positional uncertainty of the ligand from its B-factors."""
    b = [float(ln[60:66]) for ln in open(pdb_path)
         if ln.startswith("HETATM") and ln[17:20].strip() == comp and ln[60:66].strip()]
    if not b:
        return None
    b = np.array(b)
    return {"n_atoms": len(b), "b_mean": float(b.mean()), "b_max": float(b.max()),
            "u_rms": float(np.sqrt(b.mean() / (8 * np.pi**2))),
            "u_max": float(np.sqrt(b.max() / (8 * np.pi**2)))}


def rigid_perturb(Y, d, rng, mode="translate"):
    """Displace the whole ligand by d Angstrom (RMS) - translation or rotation."""
    if mode == "translate":
        v = rng.normal(size=3)
        return Y + d * v / np.linalg.norm(v)
    # rotation about the centroid, scaled so the RMS atom displacement equals d
    c = Y.mean(0)
    r = Y - c
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rms_perp = np.sqrt((np.linalg.norm(np.cross(r, axis), axis=1) ** 2).mean())
    if rms_perp < 1e-6:
        return Y.copy()
    theta = d / rms_perp
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return (r @ R.T) + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../src")
    ap.add_argument("--pdb", default="5RGE", help="HG3.17, the best-characterised catalyst")
    ap.add_argument("--comp", default="6NT", help="bound transition-state analogue")
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reps", type=int, default=6, help="random directions per displacement")
    ap.add_argument("--site-cut", type=float, default=6.0, help="A from ligand = active site")
    ap.add_argument("--far-cut", type=float, default=15.0, help="A from ligand = noise floor")
    ap.add_argument("--max-far", type=int, default=36, help="subsample far positions (cost)")
    ap.add_argument("--out", default="ts_sensitivity.json")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    sys.path.insert(0, ".")
    import torch

    from bayes_design.model import MPNNWrapper
    from bayes_design.utils import get_ligand, seq_struct_from_ligand
    from preorganization import clean_pdb

    if a.device:
        device = torch.device(a.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tag, n_lig, n_contact = clean_pdb(a.pdb, a.comp, a.cache)
    ctx = get_ligand(pdb_code=tag, structures_dir=a.cache, chain_list=("A",))
    seq, struct = seq_struct_from_ligand(ctx)
    Y0 = ctx["Y"].cpu().numpy().astype(np.float64).copy()
    print(f"{a.pdb}: L={len(seq)}  ligand atoms parsed={len(Y0)} (of {n_lig} written), "
          f"protein contacts<5A={n_contact}", flush=True)

    bf = ligand_bfactors(os.path.join(a.cache, f"{a.pdb}.pdb"), a.comp)
    if bf:
        print(f"  ligand B-factors: mean={bf['b_mean']:.1f} A^2 -> u_rms={bf['u_rms']:.2f} A, "
              f"max -> u_max={bf['u_max']:.2f} A", flush=True)

    # Positions: CA distance to the nearest ligand atom.
    ca = struct[:, 1].cpu().numpy().astype(np.float64)
    dmin = np.linalg.norm(ca[:, None] - Y0[None], axis=-1).min(1)
    site = np.where(dmin <= a.site_cut)[0]
    far = np.where(dmin >= a.far_cut)[0]
    if len(site) == 0:
        raise SystemExit("no residues within site cutoff - ligand parse likely failed")
    if len(far) > a.max_far:  # the floor only needs a sample, and each costs a forward pass
        far = far[np.linspace(0, len(far) - 1, a.max_far).astype(int)]
    print(f"  site positions (<{a.site_cut} A): {len(site)}   "
          f"far positions (>{a.far_cut} A): {len(far)}", flush=True)
    probe = np.concatenate([site, far])
    is_site = np.concatenate([np.ones(len(site), bool), np.zeros(len(far), bool)])

    holo = MPNNWrapper(device=device, model_type="ligand_mpnn",
                       checkpoint_path=os.path.join(
                           a.src, "bayes_design/ligand_mpnn/model_params/ligandmpnn_v_32_010_25.pt"),
                       context=ctx, use_ligand=True)

    order = list(range(len(seq)))

    def probs_now():
        """Per-position 20-vectors at the probe positions, current ligand coords."""
        out = np.zeros((len(probe), 20))
        for s in range(0, len(probe), a.batch):
            toks = [int(t) for t in probe[s:s + a.batch]]
            p = holo(seq=[seq] * len(toks), struct=struct.clone(), decode_order=order,
                     token_to_decode=toks, mask_type="bidirectional_mlm")
            out[s:s + len(toks)] = p.float().cpu().numpy()
        return out

    def set_Y(Ynew):
        holo.context["Y"] = torch.tensor(Ynew, dtype=torch.float32, device=ctx["Y"].device)
        holo._features = None

    base = probs_now()

    # CEILING: the ligand removed entirely.
    holo.use_ligand = False
    holo._features = None
    apo = probs_now()
    holo.use_ligand = True
    holo._features = None

    d_apo = kl_rows(base, apo)
    ceil_site, ceil_far = float(d_apo[is_site].mean()), float(d_apo[~is_site].mean())
    print(f"\n  CEILING  KL(holo||apo): site={ceil_site:.4f}  far={ceil_far:.4f} nats\n", flush=True)

    rng = np.random.default_rng(0)
    grid = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    rows = []
    for mode in ("translate", "rotate"):
        for d in grid:
            ks, kf = [], []
            for _ in range(a.reps):
                set_Y(rigid_perturb(Y0, d, rng, mode))
                p = probs_now()
                dk = kl_rows(base, p)
                ks.append(float(dk[is_site].mean()))
                kf.append(float(dk[~is_site].mean()))
            row = {"mode": mode, "d": d,
                   "kl_site": float(np.mean(ks)), "kl_site_sd": float(np.std(ks, ddof=1)),
                   "kl_far": float(np.mean(kf)), "kl_far_sd": float(np.std(kf, ddof=1)),
                   "frac_ceiling": float(np.mean(ks) / ceil_site)}
            rows.append(row)
            print(f"  {mode:<9} d={d:>4.2f} A   KL_site={row['kl_site']:.4f}"
                  f"+/-{row['kl_site_sd']:.4f}  ({100 * row['frac_ceiling']:5.1f}% of ceiling)"
                  f"   KL_far={row['kl_far']:.4f}", flush=True)
    set_Y(Y0)

    res = {"pdb": a.pdb, "comp": a.comp, "n_ligand_atoms": int(len(Y0)),
           "n_site": int(len(site)), "n_far": int(len(far)),
           "ceiling_site": ceil_site, "ceiling_far": ceil_far,
           "bfactors": bf, "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)

    print("\n" + "=" * 76)
    print("Can a GS->TS displacement clear the noise floor and the data resolution?")
    print("=" * 76)
    tr = [r for r in rows if r["mode"] == "translate"]
    floor = max(r["kl_far"] for r in tr)
    over = [r for r in tr if r["kl_site"] > 3 * floor]
    print(f"  spatial noise floor (max KL beyond {a.far_cut:.0f} A) = {floor:.4f} nats")
    if over:
        print(f"  site signal first exceeds 3x that floor at d = {over[0]['d']:.2f} A "
              f"({100 * over[0]['frac_ceiling']:.1f}% of the ligand ceiling)")
    else:
        print("  site signal never exceeds 3x the floor at any displacement tested")
    if bf:
        print(f"  ligand's own coordinate uncertainty      u_rms = {bf['u_rms']:.2f} A")
        print("  A differential smaller than u_rms is below the resolution of the input data.")
    print("\n  Chemistry to compare against: proton transfer / partial bond elongation")
    print("  moves heavy atoms ~0.2-0.5 A; planar->tetrahedral at a carbonyl ~0.5-0.8 A.")


if __name__ == "__main__":
    main()
