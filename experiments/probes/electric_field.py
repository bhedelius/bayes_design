"""The electric field the protein projects onto the reacting bond.

This is Warshel's preorganization as a computable quantity, and the descriptor
Boxer's group measures directly by vibrational Stark spectroscopy - in
ketosteroid isomerase the field at the carbonyl tracks log k_cat at r^2 ~ 0.9.
It needs only static coordinates and force-field partial charges: no MD, no QM.

For the Kemp eliminase series the geometry is unambiguous. In every structure
Asp127 sits 1.55 A from the ligand (the catalytic base) and residue 50 sits 2.33 A
from ligand atom N1 - N1 is the atom that carries the developing negative charge,
and residue 50 is the oxyanion hole. So the descriptor is the field and potential
at N1.

Prediction: a more positive potential at N1 (better stabilization of the developing
negative charge) in the tightly-packed, more active variants (Gln/Met/Phe) than in
the water-filled ones (His/Ser/Ala/Lys).

Charges come from PDB2PQR (AMBER); the sum is evaluated in vacuum, so absolute
magnitudes are inflated, but a constant dielectric only rescales and cannot change
the ranking across variants of the same protein.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

PACKED = {"Q50(wt)": "4BS0", "Q50M": "6TU6", "Q50F": "6SRW"}
WATERY = {"Q50H": "6SRZ", "Q50S": "6SRY", "Q50A": "6SS1", "Q50K": "6SS3"}
# 332.06 converts e/A^2 to MV/cm for the field, e/A to kcal/mol/e for the potential.
COUL = 332.0637


def protein_only(pdb, cache, keep_water=False):
    """Chain A protein (optionally with waters), for charge assignment."""
    src = os.path.join(cache, f"{pdb}.pdb")
    out = os.path.join(cache, f"{pdb}_prot{'W' if keep_water else ''}.pdb")
    # pdb2pqr aborts on residues with missing backbone atoms (4BS0 has a
    # truncated Gln303), so drop incomplete residues up front.
    have = {}
    for ln in open(src):
        if ln.startswith("ATOM") and ln[21] == "A":
            have.setdefault(ln[22:27], set()).add(ln[12:16].strip())
    ok = {k for k, v in have.items() if {"N", "CA", "C", "O"} <= v}
    with open(src) as f, open(out, "w") as g:
        for ln in f:
            if ln.startswith("ATOM") and ln[22:27] not in ok:
                continue
            # Keep one conformer per atom and blank the altloc column: pdb2pqr
            # fails outright on files with alternates (4BS0 has 1028 such atoms).
            if ln.startswith("ATOM") and ln[21] == "A" and ln[16] in " A":
                g.write(ln[:16] + " " + ln[17:])
            elif (keep_water and ln.startswith("HETATM") and ln[17:20] == "HOH"
                  and ln[16] in " A"):
                g.write(ln[:16] + " " + ln[17:])
        g.write("END\n")
    return out


def run_pqr(inp, venv):
    out = inp.replace(".pdb", ".pqr")
    if not os.path.exists(out):
        subprocess.run([venv, "-m", "pdb2pqr", "--ff=AMBER", "--noopt",
                        "--keep-chain", inp, out],
                       check=True, capture_output=True, timeout=900)
    xyz, q = [], []
    for ln in open(out):
        if ln.startswith(("ATOM", "HETATM")):
            p = ln.split()
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            q.append(float(p[-2]))
    return np.array(xyz), np.array(q)


def ligand_atoms(pdb, cache):
    m = next(iter(PDBParser(QUIET=True).get_structure("x", os.path.join(cache, f"{pdb}.pdb"))))
    ch = m["A"]
    best, bn = None, -1
    prot = np.array([a.get_coord() for r in ch if r.id[0] == " " for a in r])
    for r in ch:
        if r.get_resname() != "6NT":
            continue
        d = {a.get_name(): np.array(a.get_coord()) for a in r}
        n = int((np.linalg.norm(prot[:, None] - np.array(list(d.values()))[None], axis=-1) < 5).sum())
        if n > bn:
            best, bn = d, n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="pdbs_kemp")
    ap.add_argument("--venv", default="../.venv/bin/python")
    a = ap.parse_args()

    rows = []
    for group, members in (("packed", PACKED), ("watery", WATERY)):
        for v, pdb in members.items():
            try:
                lig = ligand_atoms(pdb, a.cache)
                n1, n2 = lig["N1"], lig["N2"]
                axis = (n2 - n1) / np.linalg.norm(n2 - n1)
                row = {"variant": v, "group": group, "pdb": pdb}
                for tag, water in (("", False), ("_wat", True)):
                    xyz, q = run_pqr(protein_only(pdb, a.cache, water), a.venv)
                    d = n1 - xyz
                    r = np.linalg.norm(d, axis=1)
                    keep = r > 0.1
                    # potential at N1: positive stabilises developing negative charge
                    row[f"potential{tag}"] = float(COUL * (q[keep] / r[keep]).sum())
                    fld = COUL * (q[keep, None] * d[keep] / r[keep, None] ** 3).sum(0)
                    row[f"field_mag{tag}"] = float(np.linalg.norm(fld))
                    row[f"field_axis{tag}"] = float(fld @ axis)
                rows.append(row)
                print(f"  {v:<9} {group:<7} potential={row['potential']:+9.2f} "
                      f"|F|={row['field_mag']:8.1f}  F.axis={row['field_axis']:+9.1f}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {v} ({pdb}) FAILED: {type(e).__name__}: {e}", flush=True)

    json.dump(rows, open("field_results.json", "w"), indent=1)
    if len(rows) < 5:
        return
    print("\n" + "=" * 74)
    print("Does the field at the oxyanion atom separate active from inactive?")
    print("=" * 74)
    print(f"{'descriptor':>16} {'packed':>11} {'watery':>11} {'p':>8}  ranking (best->worst)")
    for k in ("potential", "field_mag", "field_axis",
              "potential_wat", "field_mag_wat", "field_axis_wat"):
        p = [r[k] for r in rows if r["group"] == "packed"]
        w = [r[k] for r in rows if r["group"] == "watery"]
        u = stats.mannwhitneyu(p, w)
        order = sorted(rows, key=lambda r: -r[k])
        tag = "".join("P" if r["group"] == "packed" else "w" for r in order)
        print(f"{k:>16} {np.mean(p):>11.2f} {np.mean(w):>11.2f} {u.pvalue:>8.3f}  {tag}")
    print("\n  P = packed (more active), w = water-filled (less active).")
    print("  perfect separation = PPPwwww; best attainable p with 3 vs 4 is 0.057.")
    print("  '_wat' rows include ordered crystallographic waters in the sum.")


if __name__ == "__main__":
    main()
