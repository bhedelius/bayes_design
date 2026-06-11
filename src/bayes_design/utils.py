import os

import numpy as np
import torch
from Bio.PDB import PDBParser

from .protein_mpnn.protein_mpnn_utils import StructureDatasetPDB, parse_PDB

AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWYX"


def get_protein(pdb_code="6MRR", structures_dir="./data/structures"):
    """Get a sequence in string format and 4-atom protein structure in L x 4 x 3
    tensor format (with atoms in N CA CB C order).
    """
    pdb_path = os.path.join(structures_dir, pdb_code + ".pdb")
    if not os.path.exists(pdb_path):
        os.system(f"cd {structures_dir} && wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    chain_list = ["A"]
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    protein = dataset_valid[0]
    struct = torch.tensor(
        [
            protein["coords_chain_A"]["N_chain_A"],
            protein["coords_chain_A"]["CA_chain_A"],
            protein["coords_chain_A"]["C_chain_A"],
            protein["coords_chain_A"]["O_chain_A"],
        ]
    ).transpose(0, 1)
    return protein["seq"], struct


def get_ligand(pdb_code="6MRR", structures_dir="./data/structures", chain_list=("A",), ligand_file=None):
    """Parse a protein and its bound ligand into a LigandMPNN input dict.

    Returns the dict consumed by LigandMPNN's ``featurize`` (backbone ``X``,
    sequence ``S``, ``mask``, ligand atoms ``Y``/``Y_t``/``Y_m``, residue indices,
    chain labels, plus an all-ones ``chain_mask`` which featurize requires but
    parse_PDB does not set). Protein residues come from ``chain_list`` while
    ligand atoms are taken from HETATM records across *all* chains (so a ligand
    on another chain is still captured); waters are dropped by parse_PDB. Pass
    ``ligand_file`` to merge ligand atoms from a separate PDB.
    """
    from .ligand_mpnn.data_utils import parse_PDB as parse_PDB_ligand

    pdb_path = os.path.join(structures_dir, pdb_code + ".pdb")
    if not os.path.exists(pdb_path):
        os.system(f"cd {structures_dir} && wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")

    ligand_source = pdb_path
    if ligand_file is not None:
        ligand_source = _merge_ligand_into_pdb(pdb_path, ligand_file, structures_dir, pdb_code)

    input_dict, *_ = parse_PDB_ligand(pdb_path, chains=list(chain_list))
    full_dict, *_ = parse_PDB_ligand(ligand_source, chains=[])
    for key in ("Y", "Y_t", "Y_m"):
        input_dict[key] = full_dict[key]

    input_dict["chain_mask"] = torch.ones_like(input_dict["mask"], dtype=torch.float32)
    return input_dict


def seq_struct_from_ligand(input_dict):
    """Derive (seq string, L x 4 x 3 struct) from a `get_ligand` dict.

    Ligand-aware modes must take the sequence and backbone from the same
    LigandMPNN parse that produced the ligand features, because the LigandMPNN
    and ProteinMPNN parsers can disagree on the residue set (e.g. insertion
    codes); the decode loop's sequence length must match the bound features.
    """
    seq = "".join(AMINO_ACID_ORDER[int(i)] for i in input_dict["S"].tolist())
    struct = input_dict["X"].float()  # already N, CA, C, O ordered
    return seq, struct


def _merge_ligand_into_pdb(pdb_path, ligand_file, structures_dir, pdb_code):
    """Append a separate ligand file's HETATM records onto the protein PDB so
    parse_PDB's element/coordinate logic captures the externally-supplied ligand."""
    with open(pdb_path) as f:
        protein_lines = [ln for ln in f if not ln.startswith(("END", "CONECT"))]
    # Only HETATM records contribute ligand atoms; ATOM records (protein) are
    # ignored so passing a full complex PDB does not duplicate the protein.
    with open(ligand_file) as f:
        ligand_lines = [ln for ln in f if ln.startswith("HETATM")]
    combined_path = os.path.join(structures_dir, f"{pdb_code}_with_ligand.pdb")
    with open(combined_path, "w") as f:
        f.writelines(protein_lines)
        f.writelines(ligand_lines)
        f.write("END\n")
    return combined_path


def get_fixed_position_mask(fixed_position_list, seq_len):
    # Masked positions are the positions to predict/design
    # Default to no fixed positions, and thus predict all positions
    fixed_position_mask = np.zeros(seq_len)
    # Preserve fixed positions
    for i in range(0, len(fixed_position_list), 2):
        # -1 because residues are 1-indexed
        fixed_range_start = fixed_position_list[i] - 1
        # -1 because residues are 1-indexed and +1 because we are including the endpoint
        fixed_range_end = fixed_position_list[i + 1]
        fixed_position_mask[fixed_range_start:fixed_range_end] = 1.0
    return fixed_position_mask


def get_cb_coordinates(pdb_code, structures_dir="/data/structures"):
    """Gets the coordinates of the primary four atoms for each residue. Returns an
    L x 4 x 3 array, with the atoms in the following order: N, CA, C, CB. For glycine,
    provides CA coordinates in place of CB coordinates.
    Args:
        pdbfile (str): the full file path of a protein structure file in PDB format
    Returns:
        ((L x 4 x 3) torch.Tensor): the 3D coordinates of the primary 4 atoms in each
            amino acid in the sequence
    """
    pdb_path = os.path.join(structures_dir, pdb_code + ".pdb")
    residues = list(
        PDBParser(PERMISSIVE=True, QUIET=True)
        .get_structure(id=os.path.basename(pdb_path), file=pdb_path)[0]
        .get_residues()
    )

    L = len(residues)
    cb_coordinates = np.zeros((L, 3), dtype=np.float32)

    # Set the coordinates for every residue
    for i, residue in enumerate(residues):
        try:
            if residue.resname == "GLY":
                cb_coordinates[i, :] = residue["CA"].get_coord()
            else:
                cb_coordinates[i, :] = residue["CB"].get_coord()
        except KeyError:
            cb_coordinates[i, :] = residue["Cb".lower()].get_coord()

    return torch.tensor(cb_coordinates)


def compute_distance_matrix(coordinates, epsilon=0.0):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 3) torch.Tensor): an array of all four major atom coordinates
            per residue
        epsilon (float): a term to stabilize the gradients (because backpropping through sqrt
            gives you NaN at 0)
    Returns:
        ((L x L) torch.Tensor): the distance matrix for the residues
    """
    # In reality, pred_coordinates is an output of the network, but we initialize it here for a minimal working example
    L = len(coordinates)
    gram_matrix = torch.mm(coordinates, torch.transpose(coordinates, 0, 1))
    gram_diag = torch.diagonal(gram_matrix, dim1=0, dim2=1)
    # gram_diag: L
    diag_1 = torch.matmul(gram_diag.unsqueeze(-1), torch.ones(1, L).to(coordinates.device))
    # diag_1: L x L
    diag_2 = torch.transpose(diag_1, dim0=0, dim1=1)
    # diag_2: L x L
    squared_distance_matrix = diag_1 + diag_2 - (2 * gram_matrix)
    distance_matrix = torch.sqrt(squared_distance_matrix + epsilon)
    return distance_matrix


def compute_bins(matrix, bins, include_less_than=False, include_greater_than=False):
    """Bin values based on the bins array. Works for distances and trRosetta features.
    Args:
        matrix ((L x n) torch.Tensor): the matrix to bin
        bins ((n_bins) array-like): the bin endpoints
        include_less_than (bool): whether to include a bin for less than the min value
        include_greater_than (bool): whether to include a bin for greater than the max value
    Returns:
        binned_matrix ((L x n x n_bins) torch.Tensor): the matrix, but binned
    """
    L, n = matrix.shape
    # Number of bins is based on whether we have a bin for less than the lowest and greater than the highest
    n_bins = len(bins) - 1 + include_less_than + include_greater_than

    # Populate distogram
    binned_matrix = torch.zeros((L, n, n_bins))

    if include_less_than:
        binned_matrix[:, :, 0] = matrix < bins[0]

    for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
        # Bins are shifted by one if we have a "less than" bin
        binned_matrix[:, :, include_less_than + i] = (matrix >= bin_min) * (matrix < bin_max)

    if include_greater_than:
        binned_matrix[:, :, -1] = matrix >= bins[-1]

    return binned_matrix


def compute_distogram(coordinates):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 4 x 3) np.ndarray): an array of all four major atom coordinates
            per residue
    Returns:
        ((N x L x L) torch.Tensor): the binned distance matrix for the atoms
    """
    distance_matrix = compute_distance_matrix(coordinates)
    # Make sure all distance values are positive
    assert torch.all(distance_matrix >= 0)
    # The endpoints of the bins (n_bins + 1 endpoints)
    tr_rosetta_bins = np.arange(2.5, 20.5, 0.5)
    # Compute the distogram
    distogram = compute_bins(
        matrix=distance_matrix, bins=tr_rosetta_bins, include_less_than=True, include_greater_than=True
    )
    # No need to normalize probabilities to sum to 1, because there is just one one in each distogram

    return distogram
