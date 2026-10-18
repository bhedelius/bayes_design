"""Batch-score sequences against a structure under a BayesDesign/ProteinMPNN model.

Walks a directory of FASTA/txt sequence files, scores each full sequence's
log p(struct|seq), and writes a per-sequence CSV with per-file summary stats.
Used as the selection primitive for ranking designed sequences.
"""

import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from bayes_design.decode import decode_order_dict
from bayes_design.evaluate import metric_dict
from bayes_design.model import build_model, model_dict
from bayes_design.utils import resolve_protein_input


def get_sequences_from_file(file_path):
    """Read amino-acid sequences from a FASTA or one-per-line txt file.

    FASTA records (``>`` headers) are joined across wrapped lines; a header-less file is read
    one sequence per line. Only uppercase/gap sequences are returned.
    """
    with open(file_path) as f:
        lines = [ln.strip() for ln in f]
    if any(ln.startswith(">") for ln in lines):
        sequences, current = [], []
        for ln in lines:
            if ln.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            elif ln:
                current.append(ln)
        if current:
            sequences.append("".join(current))
    else:
        sequences = [ln for ln in lines if ln]
    return [s for s in sequences if s and all(c.isupper() or c == "-" for c in s)]


def score_sequences(
    pdb_path,
    sequences_dir,
    model_name="bayes_design",
    decode_order="n_to_c",
    bayes_balance_factor=0.002,
    device=None,
    output_file=None,
):
    """Score every sequence under ``sequences_dir`` against ``pdb_path``.

    Returns a DataFrame of (File_Path, Sequence_Index, Log_Prob) and prints per-file
    mean/std/count; writes the DataFrame to ``output_file`` when given.
    """
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = SimpleNamespace(pdb_path=pdb_path, protein_id=None)
    seq, struct = resolve_protein_input(args)

    print(f"Initializing {model_name}...")
    prob_model = build_model(model_name, device=device, bayes_balance_factor=bayes_balance_factor)

    # Score the whole sequence: nothing fixed.
    fixed_position_mask = np.zeros(len(seq))
    mask_type = "bidirectional_autoregressive"
    order = decode_order_dict[decode_order](seq)

    input_path = os.path.abspath(sequences_dir)
    input_pardirname = os.path.basename(input_path)
    seq_paths = glob.glob(os.path.join(input_path, "**", "*.fasta"), recursive=True) + glob.glob(
        os.path.join(input_path, "**", "*.txt"), recursive=True
    )

    results = []
    for seq_path in seq_paths:
        print(f"Processing {seq_path}...")
        sequences = get_sequences_from_file(seq_path)
        if not sequences:
            print(f"  No sequences found in {seq_path}. Skipping.")
            continue
        for i, input_seq in enumerate(sequences):
            if len(input_seq) != len(seq):
                print(f"  Seq {i}: length mismatch ({len(input_seq)} vs {len(seq)}). Skipping.")
                continue
            score = metric_dict["log_prob"](
                seq=input_seq,
                prob_model=prob_model,
                decode_order=order,
                structure=struct,
                fixed_position_mask=fixed_position_mask,
                mask_type=mask_type,
            )
            seq_pardirname = os.path.basename(os.path.dirname(seq_path))
            seq_id = seq_pardirname if seq_pardirname != input_pardirname else os.path.basename(seq_path)
            results.append({"File_Path": seq_id, "Sequence_Index": i, "Log_Prob": round(score, 4)})

    df = pd.DataFrame(results)
    if len(df) > 0:
        summary = df.groupby("File_Path")["Log_Prob"].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["File_Path", "Mean", "Std", "Count"]
        print("\nSummary Statistics by File:")
        print(summary.to_string(index=False))
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results written to {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Score p(structure|sequence) for sequences under a structure.")
    parser.add_argument("--pdb_path", help="Path to local PDB file", required=True)
    parser.add_argument("--sequences_dir", help="Directory of FASTA/txt sequence files (searched recursively)", required=True)
    parser.add_argument("--output_file", help="Optional CSV output path", default=None)
    parser.add_argument("--model_name", default="bayes_design", choices=list(model_dict.keys()))
    parser.add_argument("--decode_order", default="n_to_c", choices=list(decode_order_dict.keys()))
    parser.add_argument("--bayes_balance_factor", default=0.002, type=float)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    return score_sequences(
        pdb_path=args.pdb_path,
        sequences_dir=args.sequences_dir,
        model_name=args.model_name,
        decode_order=args.decode_order,
        bayes_balance_factor=args.bayes_balance_factor,
        device=args.device,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
