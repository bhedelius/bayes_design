"""Assemble enzymes with annotated catalytic residues, mapped onto PDB structures.

UniProt supplies ACT_SITE positions; PDBe SIFTS maps UniProt numbering onto PDB
author numbering. Output feeds the same scoring path as the B-factor experiment,
with a per-residue `catalytic` flag.
"""

import argparse
import json
import urllib.parse
import urllib.request

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/"


def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def uniprot_entries(n):
    q = "(ft_act_site:*) AND (database:pdb) AND (reviewed:true) AND (length:[80 TO 400])"
    url = (f"{UNIPROT}?query={urllib.parse.quote(q)}"
           f"&fields=accession,ft_act_site,xref_pdb&format=json&size={min(n, 500)}")
    return get_json(url)["results"]


def act_sites(entry):
    return [f["location"]["start"]["value"] for f in entry.get("features", [])
            if f["type"] == "Active site" and f["location"]["start"]["value"] ==
            f["location"]["end"]["value"]]


def pdb_ids(entry):
    return [x["id"] for x in entry.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"
            and any(p.get("key") == "Method" and p.get("value") == "X-ray" for p in x.get("properties", []))]


def sifts_map(pdb_id, acc):
    """Return {chain: (uniprot_start, pdb_start)} segments for this accession."""
    data = get_json(SIFTS + pdb_id.lower())
    segs = {}
    for m in data.get(pdb_id.lower(), {}).get("UniProt", {}).get(acc, {}).get("mappings", []):
        segs.setdefault(m["chain_id"], []).append(
            (m["unp_start"], m["unp_end"], m["start"]["residue_number"], m["end"]["residue_number"]))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="catalytic.json")
    a = ap.parse_args()

    out, seen_pdb = [], set()
    for entry in uniprot_entries(400):
        if len(out) >= a.n:
            break
        acc = entry["primaryAccession"]
        sites = act_sites(entry)
        if len(sites) < 1:
            continue
        for pdb in pdb_ids(entry)[:3]:
            if pdb in seen_pdb:
                continue
            try:
                segs = sifts_map(pdb, acc)
            except Exception:  # noqa: BLE001
                continue
            if not segs:
                continue
            chain = sorted(segs)[0]
            mapped = []
            for s in sites:
                for us, ue, ps, pe in segs[chain]:
                    if us <= s <= ue:
                        mapped.append(ps + (s - us))
                        break
            if len(mapped) < len(sites) or not mapped:
                continue
            seen_pdb.add(pdb)
            out.append({"pdb": pdb, "chain": chain, "acc": acc, "catalytic_resnum": mapped})
            print(f"{pdb} chain {chain} ({acc}): catalytic at PDB residues {mapped}", flush=True)
            break

    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {len(out)} enzymes -> {a.out}")


if __name__ == "__main__":
    main()
