"""Calibrate the BayesDesign objective against mega-scale Delta G data.

Fits a linear map from per-term inverse-folding log-likelihoods to measured folding
stability (Tsuboyama et al. 2023), yielding the review's denominator weight ``lambda``
and inverse temperature ``beta``, and comparing candidate denominators: the XLNet
marginal ``p(seq)`` against a self-consistent backbone-ensemble partition function.
Each variant's sequence is scored, teacher-forced, on its wild-type backbone; features
are differenced within each domain so the target is ``ddG`` (mutant minus wild type),
which cancels per-domain offsets. Scoring needs model weights and a GPU; see
``experiments.md`` for commands.
"""

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from Bio.Seq import Seq

from ..evaluate import evaluate_sequence_log_prob
from ..model import BACKENDS, BackboneLibraryWrapper
from ..utils import AMINO_ACID_ORDER, get_protein, load_backbone_library

STANDARD_AA = set(AMINO_ACID_ORDER[:-1])  # 20 canonical, excluding 'X'
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _translate(dna):
    """Translate a coding DNA sequence to protein, dropping a trailing stop codon."""
    return str(Seq(dna).translate()).rstrip("*")


def _to_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# Each schema maps a CSV row to (domain_id, variant_name, aa_seq, is_wt, dG, ddG).
# ddG is None when it must be computed within the domain (dataset1).
def _row_dataset1(row):
    name = row["name"]
    domain_id = name.split(".pdb")[0]
    aa = _translate(row["dna_seq"])
    return domain_id, name, aa, name.endswith(".pdb"), _to_float(row.get("deltaG")), None


def _row_dataset2_3(row):
    return (
        row["WT_name"].split(".pdb")[0],
        row["name"],
        row.get("aa_seq", ""),
        row.get("mut_type") == "wt",
        _to_float(row.get("dG_ML")),
        _to_float(row.get("ddG_ML")),
    )


SCHEMAS = {"dataset1": _row_dataset1, "dataset2_3": _row_dataset2_3}


def _grouped_rows(csv_path, parse_row, max_domains=None):
    """Yield ``(domain_id, [parsed_rows])`` for contiguously-grouped domains.

    The Tsuboyama CSVs list each wild type followed by its variants, so a domain is
    closed as soon as the domain id changes.
    """
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        current_id, group, n_domains = None, [], 0
        for row in reader:
            parsed = parse_row(row)
            domain_id = parsed[0]
            if domain_id != current_id:
                if group:
                    yield current_id, group
                    n_domains += 1
                    if max_domains is not None and n_domains >= max_domains:
                        return
                current_id, group = domain_id, []
            group.append(parsed)
        if group and (max_domains is None or n_domains < max_domains):
            yield current_id, group


def _domain_offsets(wt_aa, pdb_seq):
    """Return ``(start, end)`` slicing the domain (matching ``pdb_seq``) out of a
    tagged construct, or ``None`` if it cannot be located and lengths differ."""
    start = wt_aa.find(pdb_seq)
    if start >= 0:
        return start, start + len(pdb_seq)
    if len(wt_aa) == len(pdb_seq):
        return 0, len(pdb_seq)
    return None


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _load_pool(structures_dir, max_structures=None):
    """Load every PDB in a directory keyed by its domain id, for leave-one-out decoys."""
    paths = sorted(Path(structures_dir).glob("*.pdb"))
    if max_structures is not None:
        paths = paths[:max_structures]
    return {path.stem: get_protein(path.stem, structures_dir=structures_dir)[1] for path in paths}


def score_variants(args):
    """Score every usable variant and return feature rows plus skip counts."""
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_model = BACKENDS["protein_mpnn"](device=device)
    ens_model = (
        BACKENDS["backbone_ensemble"](device=device, n_samples=args.n_ensemble, sigma=args.ensemble_sigma)
        if "backbone_ensemble" in args.denominators
        else None
    )
    xlnet_model = BACKENDS["xlnet"](device=device) if "xlnet" in args.denominators else None

    # The library denominator reuses the numerator's ProteinMPNN as its inner model.
    # Leave-one-out draws decoys from --structures-dir minus the target; otherwise the
    # library is a fixed external pool from --decoy-library.
    lib_model, pool, rng = None, None, random.Random(0)
    if "backbone_library" in args.denominators:
        if args.leave_one_out:
            pool = _load_pool(args.structures_dir, args.max_pool)
            lib_model = BackboneLibraryWrapper(num_model, library=[], device=device)
        else:
            library = load_backbone_library(args.decoy_library)
            lib_model = BackboneLibraryWrapper(num_model, library=library, n_samples=args.n_decoys, device=device)

    parse_row = SCHEMAS[args.schema]
    skip_keys = ("no_structure", "no_wt", "offset", "length", "nonstandard", "no_ddg", "no_decoys")
    rows, skips = [], dict.fromkeys(skip_keys, 0)

    for domain_id, group in _grouped_rows(args.csv, parse_row, args.max_domains):
        pdb_path = Path(args.structures_dir) / f"{domain_id}.pdb"
        if not pdb_path.exists():
            skips["no_structure"] += 1
            continue
        wt = next((r for r in group if r[3]), None)
        if wt is None or wt[4] is None:
            skips["no_wt"] += 1
            continue

        pdb_seq, struct = get_protein(domain_id, structures_dir=args.structures_dir)
        offsets = _domain_offsets(wt[2], pdb_seq)
        if offsets is None:
            skips["offset"] += 1
            continue
        start, end = offsets
        wt_dg = wt[4]

        if lib_model is not None and args.leave_one_out:
            candidates = [s for i, s in pool.items() if i != domain_id and s.shape[0] >= len(pdb_seq)]
            if not candidates:
                skips["no_decoys"] += 1
                continue
            rng.shuffle(candidates)
            lib_model.library = candidates[: args.n_decoys]

        variants = group if args.max_variants is None else group[: args.max_variants + 1]
        for _domain, name, aa, is_wt, dg, ddg in variants:
            dom = aa[start:end]
            if len(dom) != len(pdb_seq):
                skips["length"] += 1
                continue
            if not set(dom) <= STANDARD_AA:
                skips["nonstandard"] += 1
                continue
            measured = ddg if ddg is not None else (None if dg is None else dg - wt_dg)
            if measured is None:
                skips["no_ddg"] += 1
                continue

            record = {
                "domain": domain_id,
                "name": name,
                "is_wt": int(is_wt),
                "ddG": measured,
                "f_num": evaluate_sequence_log_prob(dom, num_model, structure=struct, mask_type=args.mask_type),
            }
            if ens_model is not None:
                record["f_ens"] = evaluate_sequence_log_prob(dom, ens_model, structure=struct, mask_type=args.mask_type)
            if xlnet_model is not None:
                record["f_xlnet"] = evaluate_sequence_log_prob(dom, xlnet_model, mask_type=args.mask_type)
            if lib_model is not None:
                record["f_lib"] = evaluate_sequence_log_prob(dom, lib_model, structure=struct, mask_type=args.mask_type)
            rows.append(record)

    return rows, skips


def _difference_within_domains(rows):
    """Subtract each domain's wild-type features, dropping domains without a wild type
    and the (trivially zero) wild-type rows themselves."""
    feature_keys = [k for k in ("f_num", "f_ens", "f_xlnet", "f_lib") if k in rows[0]]
    wt_by_domain = {r["domain"]: r for r in rows if r["is_wt"]}
    out = []
    for r in rows:
        if r["is_wt"] or r["domain"] not in wt_by_domain:
            continue
        wt = wt_by_domain[r["domain"]]
        out.append({"ddG": r["ddG"], **{k: r[k] - wt[k] for k in feature_keys}})
    return out, feature_keys


def _fit(features, target):
    """Least-squares fit ``target ~ features @ coef + intercept``; return fit stats."""
    from scipy.stats import spearmanr

    design = np.column_stack([*features, np.ones(len(target))])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "coef": coef.tolist(),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "pearson": float(np.corrcoef(pred, target)[0, 1]),
        "spearman": float(spearmanr(pred, target).statistic),
    }


def calibrate(diffed, feature_keys, denominators):
    """Fit ddG against the numerator alone and against each numerator/denominator pair."""
    target = np.array([d["ddG"] for d in diffed])
    f_num = np.array([d["f_num"] for d in diffed])
    fits = {"numerator_only": _fit([f_num], target)}
    den_key = {"backbone_ensemble": "f_ens", "xlnet": "f_xlnet", "backbone_library": "f_lib"}
    for den in denominators:
        key = den_key[den]
        if key not in feature_keys:
            continue
        fit = _fit([f_num, np.array([d[key] for d in diffed])], target)
        a, b, _c = fit["coef"]
        fit["lambda"] = -b / a if a else float("nan")
        fit["beta"] = 1.0 / a if a else float("nan")
        fits[den] = fit
    return fits


def _write_features(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_features(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["is_wt"] = int(r["is_wt"])
        for k in ("ddG", "f_num", "f_ens", "f_xlnet", "f_lib"):
            if k in r:
                r[k] = float(r[k])
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description="Calibrate the BayesDesign objective against Tsuboyama Delta G data.")
    parser.add_argument("--csv", required=True, help="Path to a Tsuboyama processed dataset CSV.")
    parser.add_argument(
        "--structures-dir", required=True, help="Directory of per-domain backbone PDBs (e.g. AlphaFold)."
    )
    parser.add_argument("--schema", choices=list(SCHEMAS), default="dataset1")
    parser.add_argument("--denominators", nargs="+", default=["backbone_ensemble", "xlnet"])
    parser.add_argument("--max-domains", type=int, default=None, help="Cap domains scored (start small to shake out).")
    parser.add_argument(
        "--max-variants", type=int, default=None, help="Cap variants per domain (excludes the wild type)."
    )
    parser.add_argument("--n-ensemble", type=int, default=8)
    parser.add_argument("--ensemble-sigma", type=float, default=0.1)
    parser.add_argument(
        "--decoy-library", default=None, help="Directory of backbone PDBs for the backbone_library denominator."
    )
    parser.add_argument("--n-decoys", type=int, default=None, help="Cap decoys averaged per variant (default: all).")
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="Draw backbone_library decoys from --structures-dir, excluding each target domain.",
    )
    parser.add_argument("--max-pool", type=int, default=None, help="Cap the leave-one-out decoy pool size.")
    parser.add_argument("--mask-type", default="bidirectional_mlm")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--features-out", default=None, help="Cache scored features here (CSV).")
    parser.add_argument("--reuse-features", action="store_true", help="Load --features-out and only re-fit.")
    parser.add_argument("--out", required=True, help="Write the calibration JSON here.")
    return parser


def main():
    args = _build_parser().parse_args()

    if args.reuse_features:
        if not args.features_out:
            sys.exit("--reuse-features requires --features-out")
        rows, skips = _read_features(args.features_out), {}
    else:
        rows, skips = score_variants(args)
        if args.features_out and rows:
            _write_features(args.features_out, rows)
    if not rows:
        sys.exit("No usable variants scored; check --csv/--structures-dir/--schema.")

    diffed, feature_keys = _difference_within_domains(rows)
    fits = calibrate(diffed, feature_keys, args.denominators)

    csv_path = Path(args.csv)
    result = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_csv": str(csv_path),
        "dataset_bytes": csv_path.stat().st_size if csv_path.exists() else None,
        "schema": args.schema,
        "structures_dir": args.structures_dir,
        "mask_type": args.mask_type,
        "n_ensemble": args.n_ensemble,
        "ensemble_sigma": args.ensemble_sigma,
        "decoy_library": args.decoy_library,
        "n_decoys": args.n_decoys,
        "leave_one_out": args.leave_one_out,
        "max_pool": args.max_pool,
        "n_variants": len(diffed),
        "n_domains": len({r["domain"] for r in rows}),
        "skips": skips,
        "numerator": "protein_mpnn",
        "fits": fits,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"n_variants": result["n_variants"], "n_domains": result["n_domains"], "fits": fits}, indent=2))


if __name__ == "__main__":
    main()
