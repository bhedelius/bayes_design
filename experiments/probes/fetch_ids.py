"""Pull a sequence-diverse (30% identity clusters) set of high-resolution X-ray
monomer entries from the RCSB search API."""

import argparse
import json
import urllib.request

URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def query(rows, start, res_max, len_range):
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal", "value": res_max}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                    "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                    "operator": "range", "value": {"from": len_range[0], "to": len_range[1]}}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.deposited_model_count", "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.nonpolymer_entity_count", "operator": "equals", "value": 0}},
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30},
            "group_by_return_type": "representatives",
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--res-max", type=float, default=2.0)
    ap.add_argument("--min-len", type=int, default=70)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--out", default="ids.txt")
    a = ap.parse_args()

    ids, start = [], 0
    while len(ids) < a.n:
        body = json.dumps(query(min(100, a.n - len(ids)), start, a.res_max, (a.min_len, a.max_len))).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        batch = [x["identifier"].split("_")[0] for x in d.get("result_set", [])]
        if not batch:
            break
        ids += batch
        start += 100
        print(f"{len(ids)}/{a.n} (of {d['total_count']} matching entities)")
    ids = list(dict.fromkeys(ids))[: a.n]
    with open(a.out, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} ids -> {a.out}")


if __name__ == "__main__":
    main()
