"""Command-line entry point for protein sequence design (``bayes-design``)."""

import argparse
import os
import random

import numpy as np
import torch

from bayes_design.decode import decode_algorithm_dict, decode_order_dict
from bayes_design.model import build_model, model_dict, objective_uses_context
from bayes_design.utils import (
    align_and_crop,
    build_aa_allowed_mask,
    get_ball_mask,
    get_fixed_position_mask,
    get_ligand,
    get_protein,
    load_config_and_merge,
    parse_design_regions_arg,
    resolve_protein_input,
    seq_struct_from_ligand,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Design a protein sequence for a backbone.")
    parser.add_argument(
        "--model_name",
        help="The model to use for protein sequence design",
        choices=list(model_dict.keys()),
        default="bayes_design",
    )
    parser.add_argument(
        "--protein_id", help="The PDB id to fetch from RCSB (alternative to --pdb_path)", default=None
    )
    parser.add_argument("--pdb_path", help="Path to a local PDB file (takes priority over --protein_id)", default=None)
    parser.add_argument(
        "--design_regions",
        help='JSON of named designable regions with optional per-region allowed/excluded amino acids, e.g. '
        '\'{"loop": {"positions": "63-96", "excluded_aas": "C"}}\'. Positions outside every region stay fixed. '
        "Overrides --fixed_positions/--exclude_aa when provided.",
        default=None,
    )
    parser.add_argument("--config", help="Path to a JSON config file whose values override CLI args", default=None)
    parser.add_argument(
        "--consider_wt_aa",
        help="Always keep the wild-type amino acid in the allowed set at each design position "
        "(overrides a region's excluded_aas for the WT residue)",
        action="store_true",
    )
    parser.add_argument(
        "--decode_order",
        help="The order to decode masked parts of the sequence",
        choices=list(decode_order_dict.keys()),
        default="n_to_c",
    )
    parser.add_argument(
        "--decode_algorithm",
        help="The algorithm used to decode masked parts of the sequence",
        choices=list(decode_algorithm_dict.keys()),
        default="beam",
    )
    parser.add_argument(
        "--fixed_positions",
        help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces. Example: 3 10 14 14 17 20",
        nargs="*",
        type=int,
        default=[],
    )
    parser.add_argument("--n_beams", help="The number of beams, if using beam search decoding", type=int, default=16)
    parser.add_argument(
        "--redesign",
        help="Whether to redesign an existing sequence, using the existing sequence as bidirectional context. Default is to design from scratch.",
        action="store_true",
    )
    parser.add_argument("--device", help="The GPU index to use", type=int, default=0)
    parser.add_argument(
        "--bayes_balance_factor",
        help="A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002",
        default=0.002,
        type=float,
    )
    parser.add_argument("--temperature", help="The temperature to use for sampling", default=1.0, type=float)
    parser.add_argument(
        "--contrast_weight",
        help="Scales the magnitude of a denominator/contrast term marked 'dose' in the objective. "
        "1.0 is the plain ratio; other values titrate how hard the contrast is pushed.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--n_ensemble",
        help="Backbone-ensemble sample count for ensemble objectives "
        "(bayes_design_basin, bayes_design_selfconsistent).",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--ensemble_sigma",
        help="Backbone-ensemble Gaussian noise std in angstroms.",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--decoy_library",
        help="Directory of backbone PDBs (decoys / AlphaFold models) for the "
        "backbone-library denominator (bayes_design_decoy); --n_ensemble caps the decoys averaged.",
        default=None,
    )
    parser.add_argument("--n_designs", help="The number of designs to generate", default=1, type=int)
    parser.add_argument("--seed", help="The random seed to use", default=0, type=int)
    parser.add_argument("--results_dir", help="The directory to save results to", default="./results")
    parser.add_argument("--exclude_aa", nargs="+", default=[])
    parser.add_argument(
        "--ligand_file",
        help="Optional PDB whose HETATM records supply the ligand atoms (ligand modes); "
        "defaults to the protein PDB's own HETATM records.",
        default=None,
    )
    parser.add_argument(
        "--decoy_id",
        help="PDB id of the decoy/competing fold for multi-state modes (e.g. fold_specificity).",
        default=None,
    )
    parser.add_argument("--ligand_file_b", help="PDB with the off-target ligand for ligand_selectivity.", default=None)
    parser.add_argument(
        "--ball_mask",
        help="Free only the residues within a ball around each designed position (instead of fixing exactly "
        "--fixed_positions). Useful for conformational-specificity design.",
        action="store_true",
    )
    return parser


def example_design(args: argparse.Namespace) -> dict:
    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available()) else "cpu")

    # Get sequence and structure of protein to redesign (local --pdb_path or RCSB --protein_id)
    seq, struct = resolve_protein_input(args)
    orig_seq = seq

    # Context-aware objectives bind parsed structures/ligands at construction. Source
    # seq/struct from the same LigandMPNN parse so residue counts stay consistent.
    contexts = {}
    if args.model_name == "fold_specificity":
        # Conformational-specificity design: score the sequence against the target
        # fold (main) relative to a competing fold (decoy). Align/crop the two
        # backbones to a common length so corresponding positions line up; each
        # fold is bound to its own term as a raw backbone.
        if args.decoy_id is None:
            raise ValueError("fold_specificity requires --decoy_id (the competing fold to design against).")
        seq_decoy, struct_decoy = get_protein(args.decoy_id)
        _, _, seq, struct, struct_decoy = align_and_crop(seq, seq_decoy, struct, struct_decoy)
        orig_seq = seq
        contexts["main"] = struct
        contexts["decoy"] = struct_decoy
    elif objective_uses_context(args.model_name):
        contexts["main"] = get_ligand(args.protein_id, ligand_file=args.ligand_file)
        seq, struct = seq_struct_from_ligand(contexts["main"])
        orig_seq = seq
        if args.decoy_id is not None:
            contexts["decoy"] = get_ligand(args.decoy_id)
        if args.ligand_file_b is not None:
            contexts["ligand_b"] = get_ligand(args.protein_id, ligand_file=args.ligand_file_b)

    prob_model = build_model(
        args.model_name,
        device=device,
        contexts=contexts,
        bayes_balance_factor=args.bayes_balance_factor,
        contrast_weight=args.contrast_weight,
        ensemble_n=args.n_ensemble,
        ensemble_sigma=args.ensemble_sigma,
        decoy_library=args.decoy_library,
    )

    if args.design_regions:
        fixed_position_mask, aa_allowed_mask = build_aa_allowed_mask(
            args.design_regions, len(seq), seq=seq, consider_wt_aa=args.consider_wt_aa
        )
    elif args.ball_mask:
        res_ids = list(range(1, len(seq) + 1))
        fixed_position_mask = get_ball_mask(fixed_position_list=args.fixed_positions, struct=struct, res_ids=res_ids)
        aa_allowed_mask = None
    else:
        fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(seq))
        aa_allowed_mask = None
    masked_seq = "".join(["-" if not fixed else char for char, fixed in zip(seq, fixed_position_mask)])

    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)

    if not args.redesign:
        seq = masked_seq

    from_scratch = not args.redesign

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.decode_algorithm in ["max_prob_decode", "combinations"]:
        designed_seqs = decode_algorithm_dict[args.decode_algorithm](
            prob_model=prob_model,
            struct=struct,
            seq=seq,
            unmasked_seq=orig_seq,
            decode_order=decode_order,
            fixed_position_mask=fixed_position_mask,
            from_scratch=from_scratch,
            temperature=args.temperature,
            n_beams=args.n_beams,
            exclude_aa=args.exclude_aa,
            aa_allowed_mask=aa_allowed_mask,
        )
    else:
        designed_seqs = []
        for _ in range(args.n_designs):
            designed_seq = decode_algorithm_dict[args.decode_algorithm](
                prob_model=prob_model,
                struct=struct,
                seq=seq,
                unmasked_seq=orig_seq,
                decode_order=decode_order,
                fixed_position_mask=fixed_position_mask,
                from_scratch=from_scratch,
                temperature=args.temperature,
                n_beams=args.n_beams,
                exclude_aa=args.exclude_aa,
                aa_allowed_mask=aa_allowed_mask,
            )
            designed_seqs.append(designed_seq)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f"{args.model_name}_{args.protein_label}_sequences.txt"), "w") as f:
        for designed in designed_seqs:
            f.write(designed + "\n")

    return {
        "Original sequence": orig_seq,
        "Masked sequence (tokens to predict are indicated by a dash)": masked_seq,
        "Designed sequence": designed_seqs,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args = load_config_and_merge(args, parser)
    args = parse_design_regions_arg(args)
    seqs = example_design(args)
    if args.n_designs > 1:
        print(f"Designs written to {args.results_dir}/{args.model_name}_{args.protein_label}_sequences.txt")
    else:
        for k, v in seqs.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
