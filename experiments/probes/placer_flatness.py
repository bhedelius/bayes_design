"""H3: does a *level* catalytic profile separate good hydrolases from bad ones?

Albery & Knowles: a maximally evolved enzyme has a free-energy profile with no
state bound much more tightly than the others. The design field optimises the
opposite - it maximises quality at each state independently. PLACER's own
published analysis (Lauko 2025, Fig. S3) reports, per reaction state, the
fraction of ensemble frames in which the key catalytic interactions are formed,
and requires that fraction to be high everywhere.

H3 says the predictive quantity is not the level but the **spread across states**.
This script measures both, on the same ensembles, so the two can be compared
directly - and the comparison is made *conditional on* the level, because a
trivially flat-and-bad profile is not what the hypothesis means.

Four states are scored per protein:

    apo   no mutation
    TI1   J16, acylation tetrahedral intermediate (methyl ester leaving group)
    AEI   J09, acyl-enzyme
    TI2   75I, deacylation tetrahedral intermediate (shipped with PLACER)

The per-state readout is the fraction of ensemble frames with the catalytic triad
intact - His NE2 to the serine Ogamma, and His ND1 to the Asp/Glu carboxylate.
That definition is identical in all four states (Ogamma is present throughout,
esterified in the intermediates), which is what makes the spread meaningful.
The oxyanion-hole occupancy is reported alongside for the three ligand states.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

STATES = [("apo", None), ("TI1", "J16"), ("AEI", "J09"), ("TI2", "75I")]
HB = 3.5  # A, heavy-atom donor-acceptor cutoff


def fetch(pdb, cache):
    path = os.path.join(cache, f"{pdb}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", path)
    return path


def clean_protein(path):
    """Rewrite a PDB into the shape PLACER's parser tolerates.

    Two failure modes, both silent-ish and both hit here:

    * PLACER asserts if a HETATM chain letter collides with a protein chain, so
      the heteroatoms are moved to a chain letter no ATOM record uses.
    * PLACER *segfaults* if the input contains no heteroatoms at all, even under
      `--no-use_sm`. So they are kept rather than stripped - they are ignored for
      prediction, but the parser needs them to exist.

    Altlocs are collapsed to the first conformer; the parser rejects alternates.
    """
    out = path.replace(".pdb", "_prot.pdb")
    if os.path.exists(out):
        return out
    prot, het, used = [], [], set()
    for ln in open(path):
        if ln.startswith("ATOM") and ln[16] in " A":
            used.add(ln[21])
            prot.append(ln[:16] + " " + ln[17:])
        elif ln.startswith("HETATM") and ln[16] in " A":
            het.append(ln[:16] + " " + ln[17:])
    free = next((c for c in "zyxwvu" if c not in used), "z")
    # Waters are dropped by PLACER's ligand parser, so a structure whose only
    # heteroatoms are waters still crashes. Park a lone chloride far outside the
    # protein: --no-use_sm ignores it, and the 600-atom active-site crop excludes
    # it, so it cannot touch the prediction.
    if not any(ln[17:20].strip() not in ("HOH", "WAT") for ln in het):
        xyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in prot])
        far = xyz.mean(0) + (xyz.max(0) - xyz.min(0)).max() + 50.0
        het.append(f"HETATM99999 CL    CL {free} 999    "
                   f"{far[0]:8.3f}{far[1]:8.3f}{far[2]:8.3f}  1.00  0.00          CL\n")
    with open(out, "w") as g:
        g.writelines(prot)
        g.writelines(ln[:21] + free + ln[22:] for ln in het)
        g.write("END\n")
    return out


def read_atoms(path, model=None):
    """-> list of (chain, resnum, resname, atomname, xyz), first model only by default."""
    out, cur = [], 0
    for ln in open(path):
        if ln.startswith("MODEL"):
            cur = int(ln.split()[1])
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if model is not None and cur != model:
            continue
        if ln[16] not in " A":  # single altloc
            continue
        out.append((ln[21], ln[22:27].strip(), ln[17:20].strip(), ln[12:16].strip(),
                    np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return out


def find_triad(atoms):
    """Locate a Ser-His-Asp/Glu catalytic triad by its geometric signature."""
    def sel(res, name):
        return [a for a in atoms if a[2] == res and a[3] == name]

    best = None
    for ser in sel("SER", "OG"):
        for ne2 in sel("HIS", "NE2"):
            d1 = np.linalg.norm(ser[4] - ne2[4])
            if d1 > 3.6:
                continue
            for nd1 in sel("HIS", "ND1"):
                if (nd1[0], nd1[1]) != (ne2[0], ne2[1]):
                    continue
                for acid in atoms:
                    if acid[2] not in ("ASP", "GLU") or acid[3] not in ("OD1", "OD2", "OE1", "OE2"):
                        continue
                    d2 = np.linalg.norm(nd1[4] - acid[4])
                    if d2 > 3.6:
                        continue
                    score = d1 + d2
                    if best is None or score < best[0]:
                        best = (score, (ser[0], ser[1]), (ne2[0], ne2[1]), (acid[0], acid[1]), d1, d2)
    return best


def run_placer(placer_dir, venv, ifile, odir, suffix, ser, code, nsamples, gpu):
    # --target_res is required in EVERY state, not just apo. PLACER crops ~600
    # atoms and, without it, centres the crop on whatever heteroatom it finds -
    # which for a structure whose only het is a distant ion puts the crop nowhere
    # near the active site. That failure is silent: the run exits 0 and the
    # ensemble simply does not contain the triad.
    cmd = [venv, os.path.join(placer_dir, "run_PLACER.py"), "--ifile", ifile,
           "--odir", odir, "--suffix", suffix, "-n", str(nsamples),
           "--target_res", f"{ser[0]}-{ser[1]}"]
    if code:
        cmd += ["--mutate", f"{ser[1]}{ser[0]}:{code}", "--residue_json",
                os.path.join(placer_dir, "examples", "ligands", f"{code}.json"), "--no-use_sm"]
    # Inherit CUDA_VISIBLE_DEVICES from the caller when sharding across GPUs;
    # overriding it here would pin every shard to the same physical device.
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=placer_dir, env=env, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().split("\n")[-1][:200])
    base = os.path.splitext(os.path.basename(ifile))[0]
    return os.path.join(odir, f"{base}_{suffix}_model.pdb")


def measure(ens_pdb, triad):
    """Per-frame: is the triad intact, and is the oxyanion hole occupied?"""
    _, ser, his, acid, _, _ = triad
    nmodels = sum(1 for ln in open(ens_pdb) if ln.startswith("MODEL"))
    tri, oxy, missing = [], [], 0
    for m in range(1, nmodels + 1):
        at = read_atoms(ens_pdb, model=m)
        idx = {}
        for a in at:
            idx[(a[0], a[1], a[3])] = a[4]
        og = idx.get((ser[0], ser[1], "OG"))
        ne2 = idx.get((his[0], his[1], "NE2"))
        nd1 = idx.get((his[0], his[1], "ND1"))
        acids = [v for (c, r, n), v in idx.items()
                 if (c, r) == acid and n in ("OD1", "OD2", "OE1", "OE2")]
        if og is None or ne2 is None or nd1 is None or not acids:
            # Absent triad atoms mean the crop missed the active site, not that
            # the interaction is broken. Counted, and raised on below - scoring
            # these as "not formed" is how a cropping bug masquerades as a result.
            missing += 1
            tri.append(False)
            oxy.append(False)
            continue
        d1 = np.linalg.norm(og - ne2)
        d2 = min(np.linalg.norm(nd1 - x) for x in acids)
        tri.append(bool(d1 <= HB and d2 <= HB))
        oac = idx.get((ser[0], ser[1], "OAC"))
        if oac is None:
            oxy.append(np.nan)
        else:
            donors = [v for (c, r, n), v in idx.items()
                      if n == "N" and np.linalg.norm(v - oac) <= HB]
            oxy.append(len(donors) >= 2)
    if missing > 0.2 * nmodels:
        raise RuntimeError(f"triad atoms absent from {missing}/{nmodels} frames - "
                           "crop missed the active site")
    with np.errstate(invalid="ignore"):
        oxyf = float(np.nanmean(np.array(oxy, dtype=float))) if any(
            not isinstance(v, float) or not np.isnan(v) for v in oxy) else float("nan")
    return float(np.mean(tri)), oxyf, nmodels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="json: [{pdb, tier, label}]")
    ap.add_argument("--placer", default=os.path.expanduser("~/PLACER"))
    ap.add_argument("--venv", default=os.path.expanduser("~/placer_venv/bin/python"))
    ap.add_argument("--cache", default="pdbs_hydrolase")
    ap.add_argument("--odir", default="placer_out")
    ap.add_argument("-n", "--nsamples", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=None,
                    help="omit when sharding: CUDA_VISIBLE_DEVICES is inherited")
    ap.add_argument("--out", default="placer_flatness.json")
    ap.add_argument("--dry", action="store_true", help="triad detection only, no PLACER")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.odir, exist_ok=True)

    targets = json.load(open(a.targets))
    targets = [t for i, t in enumerate(targets) if i % a.nshards == a.shard]
    rows = []
    for t in targets:
        try:
            path = clean_protein(fetch(t["pdb"], a.cache))
            triad = find_triad(read_atoms(path))
            if triad is None:
                print(f"  {t['pdb']:<6} {t['tier']:<9} SKIP: no Ser-His-Asp triad found", flush=True)
                continue
            _, ser, his, acid, d1, d2 = triad
            print(f"  {t['pdb']:<6} {t['tier']:<9} triad Ser {ser[0]}{ser[1]} / "
                  f"His {his[0]}{his[1]} / {acid[0]}{acid[1]}  "
                  f"(OG-NE2 {d1:.2f}, ND1-acid {d2:.2f})", flush=True)
            row = {"pdb": t["pdb"], "tier": t["tier"], "label": t.get("label", t["pdb"]),
                   "ser": f"{ser[0]}{ser[1]}"}
            if a.dry:
                continue
            for name, code in STATES:
                ens = run_placer(a.placer, a.venv, os.path.abspath(path), os.path.abspath(a.odir),
                                 f"{t['pdb']}_{name}", ser, code, a.nsamples, a.gpu)
                frac, oxyf, nm = measure(ens, triad)
                row[f"{name}_triad"] = frac
                row[f"{name}_oxy"] = oxyf
                print(f"      {name:<4} triad={frac:.3f}  oxyanion={oxyf:.3f}  ({nm} frames)",
                      flush=True)
            vals = [row[f"{s}_triad"] for s, _ in STATES]
            row["level"] = float(np.mean(vals))
            row["spread"] = float(np.std(vals, ddof=1))
            row["minstate"] = float(np.min(vals))
            rows.append(row)
            json.dump(rows, open(a.out, "w"), indent=1)
        except Exception as e:  # noqa: BLE001
            print(f"  {t['pdb']:<6} FAILED: {type(e).__name__}: {e}", flush=True)

    if not rows:
        return
    print("\n" + "=" * 78)
    print("Level vs spread, by tier")
    print("=" * 78)
    print(f"{'label':<16}{'tier':<10}{'apo':>7}{'TI1':>7}{'AEI':>7}{'TI2':>7}"
          f"{'level':>8}{'spread':>8}{'min':>7}")
    for r in sorted(rows, key=lambda r: r["tier"]):
        print(f"{r['label'][:15]:<16}{r['tier']:<10}"
              f"{r['apo_triad']:>7.3f}{r['TI1_triad']:>7.3f}{r['AEI_triad']:>7.3f}"
              f"{r['TI2_triad']:>7.3f}{r['level']:>8.3f}{r['spread']:>8.3f}{r['minstate']:>7.3f}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  (n={len(rows)}); analyse with analyze_flatness.py")


if __name__ == "__main__":
    main()
