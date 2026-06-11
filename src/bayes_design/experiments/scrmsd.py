"""Structure- and sequence-comparison toolkit for evaluating designs.

These are the self-contained scoring primitives behind self-consistency RMSD
(scRMSD) evaluation: superpose two backbones (Kabsch) and measure RMSD, and
measure sequence identity between (aligned) sequences. They have no heavy
dependencies and are unit-tested.

The full scRMSD *pipeline* — predict a structure for each designed sequence with
ESMFold and filter by agreement with the target — lives on the `csdesign` branch
and additionally requires ESMFold, CATH data files, and that branch's residue-id
based masking utilities. Port that separately if/when those are wired up.
"""

import numpy as np
import torch
from Bio import pairwise2


def kabsch_alignment(target_coords, mobile_coords, coords_to_apply=None):
    """Superpose ``mobile_coords`` onto ``target_coords`` (Kabsch) and return the
    rotated/translated coordinates. If ``coords_to_apply`` is given, the rotation
    fit on the (target, mobile) pair is applied to those coordinates instead."""
    assert target_coords.shape == mobile_coords.shape, "Input coordinate arrays must have the same shape."
    assert target_coords.shape[1] == 3, "Coordinate arrays must have shape N x 3."

    centroid1 = np.mean(target_coords, axis=0)
    centroid2 = np.mean(mobile_coords, axis=0)
    target_coords_centered = target_coords - centroid1
    mobile_coords_centered = mobile_coords - centroid2

    covariance_matrix = np.dot(mobile_coords_centered.T, target_coords_centered)
    U, _, Vt = np.linalg.svd(covariance_matrix)
    d = np.linalg.det(np.dot(U, Vt))
    rotation_matrix = np.dot(U, np.dot(np.diag([1, 1, d]), Vt))

    if coords_to_apply is not None:
        mobile_coords_centered = coords_to_apply - centroid2

    coords2_aligned = np.dot(mobile_coords_centered, rotation_matrix)
    coords2_aligned += centroid1
    return coords2_aligned


def calculate_rmsd(chain_residues, new_chain_residues):
    """RMSD between two chains' CA atoms after Kabsch superposition.

    Args take lists of Bio.PDB residues (each indexable by ``"CA"``).
    """
    chain_coords = np.array([residue["CA"].get_coord() for residue in chain_residues])
    new_chain_coords = np.array([residue["CA"].get_coord() for residue in new_chain_residues])
    new_chain_coords_aligned = kabsch_alignment(chain_coords, new_chain_coords)
    return np.sqrt(np.mean(np.sum((chain_coords - new_chain_coords_aligned) ** 2, axis=1)))


def compute_rmsd(coords1, coords2, mask=None):
    """RMSD between two ``N x 3`` coordinate tensors over the masked positions,
    superposing on the complementary (unmasked) positions. NaN rows are dropped."""
    motif_mask = torch.tensor(mask).bool()
    nan_position_mask = torch.isnan(coords1.sum(-1)) | torch.isnan(coords2.sum(-1))
    coords1, coords2, motif_mask = (
        coords1[~nan_position_mask],
        coords2[~nan_position_mask],
        motif_mask[~nan_position_mask],
    )
    coords2 = torch.tensor(
        kabsch_alignment(
            target_coords=coords1[~motif_mask].numpy(),
            mobile_coords=coords2[~motif_mask].numpy(),
            coords_to_apply=coords2.numpy(),
        )
    )
    coords1 = coords1[motif_mask]
    coords2 = coords2[motif_mask]
    diff = coords1 - coords2
    return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=1))).item()


def align_sequences(seq1, seq2):
    """Global pairwise alignment; returns the highest-scoring alignment tuple."""
    alignments = pairwise2.align.globalxx(seq1, seq2)
    return max(alignments, key=lambda x: x[2])


def calculate_identity(align1, align2):
    """Percent identity between two equal-length aligned sequences."""
    assert len(align1) == len(align2)
    matches = sum(res1 == res2 for res1, res2 in zip(align1, align2))
    return matches / len(align1) * 100
