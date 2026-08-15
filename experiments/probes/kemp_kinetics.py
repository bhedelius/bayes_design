"""Assemble Kemp eliminase sequences with measured kinetics, for testing whether
kcat predictors generalise to designed enzymes (which have no natural homologs)."""
import json, os, urllib.request

# Measured values. HG3/HG3.17/HG3.R5 from PMC11581979 (Table 1); the rest are
# ordinal only - the lineage order is certain, the absolute numbers are not.
MEASURED = {  # pdb: (label, kcat s^-1, kcat/KM M^-1 s^-1)
    "5RGA": ("HG3",      6.5,   4.1e2),
    "5RGE": ("HG3.17",   604.0, 1.5e5),
    "8RD5": ("HG3.R5",   702.0, 1.7e5),
}
ORDINAL = {  # pdb: (label, rank within its lineage)
    "5RGB": ("HG3.3b", 1), "5RGC": ("HG3.7", 2), "5RGD": ("HG3.14", 3),
    "4Z08": ("KE07 r0", 0), "5D33": ("KE07 r7", 2),
    "3UXD": ("KE59 R1", 0), "3UZ5": ("KE59 R13", 1),
    "4BS0": ("HG3.17 Q50wt", 4), "6SRW": ("Q50F", 4), "6TU6": ("Q50M", 4),
    "6SRZ": ("Q50H", 3), "6SRY": ("Q50S", 3), "6SS1": ("Q50A", 3), "6SS3": ("Q50K", 3),
}
AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
SUBSTRATE = "c1cc2c(cc1[N+](=O)[O-])cno2"   # 5-nitrobenzisoxazole (PDB H5J)

def seqres(pdb, cache="pdbs_kemp"):
    """Full construct sequence from SEQRES (not ATOM: avoids disordered gaps)."""
    path = os.path.join(cache, f"{pdb}.pdb")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", path)
    seq = []
    for ln in open(path):
        if ln.startswith("SEQRES") and ln[11] == "A":
            seq += [AA3.get(t, "") for t in ln[19:].split()]
    return "".join(seq)

if __name__ == "__main__":
    rows = []
    for pdb, (lab, kcat, kkm) in MEASURED.items():
        rows.append({"pdb": pdb, "label": lab, "kcat": kcat, "kcat_km": kkm,
                     "rank": None, "sequence": seqres(pdb), "SMILES": SUBSTRATE})
    for pdb, (lab, rank) in ORDINAL.items():
        rows.append({"pdb": pdb, "label": lab, "kcat": None, "kcat_km": None,
                     "rank": rank, "sequence": seqres(pdb), "SMILES": SUBSTRATE})
    for r in rows:
        print(f"{r['pdb']}  {r['label']:<14} L={len(r['sequence']):>3}  "
              f"kcat={r['kcat'] if r['kcat'] else '-':>6}")
    json.dump(rows, open("kemp_kinetics.json", "w"), indent=1)
    import csv
    with open("catpred_input.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["SMILES", "sequence", "pdbpath"])
        for r in rows: w.writerow([r["SMILES"], r["sequence"], r["pdb"]])
    print(f"\nwrote {len(rows)} entries -> kemp_kinetics.json, catpred_input.csv")
    ident = sum(a == b for a, b in zip(rows[0]["sequence"], rows[1]["sequence"]))
    print(f"HG3 vs HG3.17 identity: {ident}/{len(rows[0]['sequence'])} "
          f"= {100*ident/len(rows[0]['sequence']):.1f}%  (they differ by only a few mutations)")
