"""Build the missing serine-hydrolase reaction states as PLACER residue definitions.

PLACER models covalent reaction intermediates as non-canonical residues grafted
onto the catalytic serine (`--mutate <res>:<code> --residue_json <file>`). The
repo ships exactly one, `75I`. Parsing its connectivity:

    C1A bonded to OG, OAC, O17, C18   -> tetrahedral carbon
    OAC has one bond                  -> the oxyanion, into the oxyanion hole
    O17 carries HW                    -> a hydroxyl, i.e. water-derived
    C18-C19-C20                       -> propyl, so the acyl group is butyryl

A tetrahedral carbon bearing Ser-O, an oxyanion and a *hydroxyl* is the
**deacylation** intermediate: 75I is TI2, not TI1.

Testing H3 (Albery-Knowles landscape flatness) needs the ligand-bound states
spread across the cycle, so this script derives the other two by editing the
embedded SDF:

  AEI  acyl-enzyme     delete O17 and its hydrogen; C1A=OAC becomes a double
                       bond. Pure deletion, so no coordinates are invented.
  TI1  acylation TI    replace O17's hydrogen with a methyl, modelling the
                       ester's departing alcohol. Adds atoms, so the added
                       coordinates are idealised (see `_place_methyl`).

Together with an unmutated apo run that gives four states, matching the set the
paper's PLACER analysis used (apo, TI1, AEI, TI2).
"""

import argparse
import json
import os

import numpy as np

# PLACER asserts the residue code is absent from its 39k-entry component database,
# so the chemically obvious names (AEI, TI1) are taken. These two are free.
CODE = {"AEI": "J09", "TI1": "J16"}


def parse_sdf(sdf):
    """-> (header lines, atoms as [x, y, z, element, rest], bonds as [i, j, order, rest])."""
    lines = sdf.split("\n")
    na, nb = int(lines[3][0:3]), int(lines[3][3:6])
    atoms, bonds = [], []
    for ln in lines[4:4 + na]:
        atoms.append([float(ln[0:10]), float(ln[10:20]), float(ln[20:30]),
                      ln[31:34].strip(), ln[34:]])
    for ln in lines[4 + na:4 + na + nb]:
        bonds.append([int(ln[0:3]), int(ln[3:6]), int(ln[6:9]), ln[9:]])
    return lines[:3], atoms, bonds


def write_sdf(header, atoms, bonds, name):
    out = [name, header[1], header[2],
           f"{len(atoms):3d}{len(bonds):3d}  0  0  1  0  0  0  0  0999 V2000"]
    for x, y, z, el, rest in atoms:
        out.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {el:<3s}{rest}")
    for i, j, o, rest in bonds:
        out.append(f"{i:3d}{j:3d}{o:3d}{rest}")
    out += ["M  END", "$$$$", ""]
    return "\n".join(out)


def drop_atoms(atoms, bonds, meta, drop):
    """Remove atom indices (1-based) and renumber the bond table."""
    keep = [i for i in range(1, len(atoms) + 1) if i not in drop]
    remap = {old: new for new, old in enumerate(keep, start=1)}
    new_atoms = [atoms[i - 1] for i in keep]
    new_bonds = [[remap[i], remap[j], o, r] for i, j, o, r in bonds
                 if i not in drop and j not in drop]
    new_meta = {k: [v[i - 1] for i in keep] for k, v in meta.items()}
    return new_atoms, new_bonds, new_meta, remap


def _place_methyl(atoms, o_idx, anchor_idx):
    """Idealised methyl carbon on `o_idx`, opposite its bond to `anchor_idx`."""
    o = np.array(atoms[o_idx - 1][:3], float)
    a = np.array(atoms[anchor_idx - 1][:3], float)
    d = o - a
    d /= np.linalg.norm(d)
    c = o + 1.43 * d  # C-O single bond
    # three hydrogens on a cone about the O-C axis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ d) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    hs = []
    for k in range(3):
        th = 2 * np.pi * k / 3
        dirn = 0.33 * d + 0.94 * (np.cos(th) * u + np.sin(th) * v)
        hs.append(c + 1.09 * dirn / np.linalg.norm(dirn))
    return c, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="75I.json", help="the shipped TI2 definition")
    ap.add_argument("--outdir", default="states")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    src = json.load(open(a.src))
    code = next(iter(src))
    rec = src[code]
    header, atoms, bonds = parse_sdf(rec["sdf"])
    meta = {k: list(rec[k]) for k in ("atom_id", "leaving", "pdbx_align")}
    idx = {name: i + 1 for i, name in enumerate(meta["atom_id"])}
    print(f"{code}: {len(atoms)} atoms, {len(bonds)} bonds")
    print(f"  C1A neighbours: "
          f"{[meta['atom_id'][(j if i == idx['C1A'] else i) - 1] for i, j, *_ in bonds if idx['C1A'] in (i, j)]}")

    written = []

    # ---- AEI: drop the hydroxyl (O17 + HW), make C1A=OAC a carbonyl ----------
    drop = {idx["O17"], idx["HW"]}
    at, bd, mt, remap = drop_atoms(atoms, bonds, meta, drop)
    c1a, oac = remap[idx["C1A"]], remap[idx["OAC"]]
    for b in bd:
        if {b[0], b[1]} == {c1a, oac}:
            b[2] = 2
            break
    else:
        raise SystemExit("C1A-OAC bond not found")
    c = CODE["AEI"]
    path = os.path.join(a.outdir, f"{c}.json")
    json.dump({c: {"sdf": write_sdf(header, at, bd, c), "atom_id": mt["atom_id"],
                   "leaving": mt["leaving"], "pdbx_align": mt["pdbx_align"]}},
              open(path, "w"))
    print(f"  AEI ({c}) -> {len(at)} atoms, {len(bd)} bonds (C1A=OAC carbonyl)  {path}")
    written.append(path)

    # ---- TI1: swap O17's hydrogen for a methyl (the departing alcohol) -------
    at = [list(x) for x in atoms]
    bd = [list(b) for b in bonds]
    mt = {k: list(v) for k, v in meta.items()}
    cm, hs = _place_methyl(at, idx["O17"], idx["C1A"])
    at, bd, mt, remap = drop_atoms(at, bd, mt, {idx["HW"]})
    o17 = remap[idx["O17"]]
    at.append([cm[0], cm[1], cm[2], "C", "  0  0  0  0  0  0  0  0  0  0  0  0"])
    mt["atom_id"].append("CM")
    mt["leaving"].append(False)
    mt["pdbx_align"].append(0)
    cm_i = len(at)
    bd.append([o17, cm_i, 1, "  0  0  0  0"])
    for k, h in enumerate(hs):
        at.append([h[0], h[1], h[2], "H", "  0  0  0  0  0  0  0  0  0  0  0  0"])
        mt["atom_id"].append(f"HM{k + 1}")
        mt["leaving"].append(False)
        mt["pdbx_align"].append(0)
        bd.append([cm_i, len(at), 1, "  0  0  0  0"])
    c = CODE["TI1"]
    path = os.path.join(a.outdir, f"{c}.json")
    json.dump({c: {"sdf": write_sdf(header, at, bd, c), "atom_id": mt["atom_id"],
                   "leaving": mt["leaving"], "pdbx_align": mt["pdbx_align"]}},
              open(path, "w"))
    print(f"  TI1 ({c}) -> {len(at)} atoms, {len(bd)} bonds (O17-methyl leaving group)  {path}")
    written.append(path)

    print(f"\nStates: apo (no --mutate), TI1={CODE['TI1']}, AEI={CODE['AEI']}, "
          f"TI2=75I (shipped).")


if __name__ == "__main__":
    main()
