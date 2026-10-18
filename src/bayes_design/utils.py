import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from Bio import pairwise2
from Bio.PDB import PDBParser

from .protein_mpnn.protein_mpnn_utils import StructureDatasetPDB, parse_PDB

AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AMINO_ACID_SET = set(AMINO_ACID_ORDER[:-1])  # 20 standard amino acids (excludes 'X')


def get_protein(pdb_code=None, pdb_path=None, structures_dir="./data/structures"):
    """Get a sequence (str) and 4-atom backbone (L x 4 x 3 tensor, N CA C O order).

    Provide a local ``pdb_path``, or a ``pdb_code`` to fetch from RCSB.
    """
    if pdb_path is not None:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: '{pdb_path}'")
        resolved_path = pdb_path
    elif pdb_code is not None:
        resolved_path = os.path.join(structures_dir, pdb_code + ".pdb")
        if not os.path.exists(resolved_path):
            os.makedirs(structures_dir, exist_ok=True)
            os.system(f"cd {structures_dir} && wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    else:
        raise ValueError("Either pdb_code or pdb_path must be provided.")
    chain_list = ["A"]
    pdb_dict_list = parse_PDB(resolved_path, input_chain_list=chain_list)
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


def sample_backbone_noise(struct, n_samples, sigma, generator=None):
    """Return ``n_samples`` perturbed copies of an (L, 4, 3) backbone.

    Adds i.i.d. Gaussian noise (std ``sigma`` angstroms) to every atom; ``sigma=0``
    returns exact copies. NaN (missing) coordinates stay NaN. Shape (n_samples, L, 4, 3).
    """
    copies = struct.unsqueeze(0).expand(n_samples, *struct.shape)
    if sigma == 0:
        return copies.clone()
    noise = torch.randn(copies.shape, generator=generator, device=struct.device, dtype=struct.dtype)
    return copies + noise * sigma


def _anm_modes(ca, cutoff, n_modes):
    """Low-frequency anisotropic-network-model modes of a CA trace.

    Returns ``(eigenvalues, displacement_vectors)`` for the ``n_modes`` softest internal
    modes, excluding the six rigid-body modes. ``displacement_vectors`` is (n_modes, m, 3).
    """
    m = ca.shape[0]
    diff = ca[None, :, :] - ca[:, None, :]
    dist2 = (diff**2).sum(-1)
    within = (dist2 <= cutoff**2) & (dist2 > 0)
    hessian = np.zeros((3 * m, 3 * m))
    for i, j in zip(*np.where(within)):
        block = -np.outer(diff[i, j], diff[i, j]) / dist2[i, j]
        hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = block
        hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
    eigvals, eigvecs = np.linalg.eigh(hessian)
    sel = slice(6, 6 + n_modes)
    return eigvals[sel], eigvecs[:, sel].T.reshape(-1, m, 3)


def sample_backbone_normal_modes(struct, n_samples, rmsd=0.5, cutoff=12.0, n_modes=10, generator=None):
    """Return ``n_samples`` copies of an (L, 4, 3) backbone deformed along elastic-network
    normal modes, scaled to a target CA-RMSD (angstroms).

    Collective, on-manifold backbone fluctuations from an anisotropic network model of the
    CA trace (soft modes weighted by ``1/sqrt(lambda)``), unlike isotropic noise. Each
    residue's four atoms translate with its CA displacement. NaN coordinates stay NaN.
    """
    copies = struct.unsqueeze(0).expand(n_samples, *struct.shape).clone()
    if rmsd == 0:
        return copies
    ca = struct[:, 1, :].detach().cpu().numpy().astype(np.float64)
    finite = np.isfinite(ca).all(axis=1)
    eigvals, modes = _anm_modes(ca[finite], cutoff, min(n_modes, 3 * finite.sum() - 6))

    gen = np.random.default_rng(None if generator is None else int(generator.initial_seed()))
    weights = 1.0 / np.sqrt(np.clip(eigvals, 1e-6, None))
    for s in range(n_samples):
        u = np.einsum("k,kmd->md", gen.standard_normal(len(eigvals)) * weights, modes)
        u *= rmsd / np.sqrt((u**2).sum(axis=1).mean())
        disp = torch.zeros_like(struct[:, 0, :])
        disp[torch.from_numpy(finite)] = torch.from_numpy(u).to(struct.dtype)
        copies[s] += disp[:, None, :]
    return copies


def load_backbone_library(structures_dir, max_structures=None):
    """Load every PDB in a directory as a list of (L, 4, 3) backbone tensors.

    A genuinely diverse structural background (decoys / AlphaFold models) for a
    self-consistent partition-function denominator, unlike noise around one backbone.
    Entries keep their native length; callers crop to the query length.
    """
    paths = sorted(Path(structures_dir).glob("*.pdb"))
    if max_structures is not None:
        paths = paths[:max_structures]
    return [get_protein(path.stem, structures_dir=structures_dir)[1] for path in paths]


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


def get_fixed_position_mask_by_res_id(fixed_position_list, res_ids):
    """Fixed-position mask keyed by residue id rather than sequence index.

    Unlike ``get_fixed_position_mask``, this handles structures with non-contiguous
    or gapped residue numbering: ``fixed_position_list`` is a flat list of
    (start, end) residue-id pairs, and a position is fixed when its ``res_ids``
    entry falls in one of those inclusive ranges.
    """
    fixed_ids = set()
    for i in range(0, len(fixed_position_list), 2):
        fixed_ids.update(range(fixed_position_list[i], fixed_position_list[i + 1] + 1))
    return np.array([1.0 if res_id in fixed_ids else 0.0 for res_id in res_ids])


def structure_to_aligned_structure(struct, aligned_seq):
    """Expand ``struct`` onto the gapped ``aligned_seq``, inserting NaN coordinates
    at gap ('-') positions so the structure lines up with the alignment."""
    new_struct = torch.full((len(aligned_seq), 4, 3), float("nan"))
    unaligned_idx = 0
    for i, char in enumerate(aligned_seq):
        if char != "-":
            new_struct[i] = struct[unaligned_idx]
            unaligned_idx += 1
    return new_struct


def align_and_crop(seq_pro, seq_anti, struct_pro, struct_anti):
    """Align two sequences/structures, crop to their shared span, and merge.

    Assumes the sequences differ only by residues present in one but not the
    other. Returns the two gapped sequences, a merged sequence (preferring
    ``seq_pro``, 'X' where both gap), and both structures expanded onto the
    cropped alignment.
    """
    alignments = pairwise2.align.globalxx(seq_pro, seq_anti, penalize_extend_when_opening=True)
    aligned_seq_pro_tmp, aligned_seq_anti_tmp, score, start, end = alignments[0]

    aligned_seq_pro = ""
    aligned_seq_anti = ""
    for i in range(len(aligned_seq_pro_tmp)):
        # Drop positions where both sequences gap
        if aligned_seq_pro_tmp[i] == "-" and aligned_seq_anti_tmp[i] == "-":
            continue
        aligned_seq_pro += aligned_seq_pro_tmp[i]
        aligned_seq_anti += aligned_seq_anti_tmp[i]

    new_struct_pro = structure_to_aligned_structure(struct_pro, aligned_seq_pro)
    new_struct_anti = structure_to_aligned_structure(struct_anti, aligned_seq_anti)

    # Strip beginning/end residues absent in the pro sequence
    n_beg = 0
    while aligned_seq_pro[n_beg] == "-":
        n_beg += 1
    n_end = 0
    while aligned_seq_pro[-(n_end + 1)] == "-":
        n_end += 1
    end = len(aligned_seq_pro) - n_end

    aligned_seq_pro = aligned_seq_pro[n_beg:end]
    aligned_seq_anti = aligned_seq_anti[n_beg:end]
    new_struct_pro = new_struct_pro[n_beg:end]
    new_struct_anti = new_struct_anti[n_beg:end]

    # Merge, preferring the pro residue and falling back to 'X' where both gap
    merged_seq = ""
    for i in range(len(aligned_seq_pro)):
        if aligned_seq_pro[i] != "-":
            merged_seq += aligned_seq_pro[i]
        elif aligned_seq_anti[i] != "-":
            merged_seq += aligned_seq_anti[i]
        else:
            merged_seq += "X"

    return aligned_seq_pro, aligned_seq_anti, merged_seq, new_struct_pro, new_struct_anti


def get_ball_mask(fixed_position_list, struct, res_ids, radius=8):
    """Mask (0) every residue within ``radius`` angstroms (CA-CA) of any unfixed
    position, keeping the rest at 1. Used to free a local ball around designed sites.

    Args:
        fixed_position_list (list): Flat (start, end) residue-id pairs for fixed positions.
        struct ((L x 4 x 3) np.array): Backbone coordinates of the protein.
        res_ids (list): Residue ids aligned to ``struct``.
        radius (float): Neighborhood radius in angstroms.
    """
    struct_ca = np.asarray(struct[:, 1], dtype=float)  # (L, 3)
    unfixed = get_fixed_position_mask_by_res_id(fixed_position_list, res_ids) == 0
    # Pairwise CA-CA distances; free (0) any position within radius of an unfixed
    # one. NaN gap rows give NaN distances (never < radius), as the old loop did.
    dists = np.linalg.norm(struct_ca[:, None, :] - struct_ca[None, :, :], axis=-1)
    ball_mask = np.ones(len(res_ids))
    ball_mask[(dists[unfixed] < radius).any(axis=0)] = 0
    return ball_mask


####################################################################################
# Per-position amino-acid masks and design-region constraints
####################################################################################


def aa_allowed_mask_from_exclude(exclude_aa, seq_len):
    """Build an (seq_len, 21) mask allowing all 20 standard AAs except ``exclude_aa``.

    Lets the legacy global ``exclude_aa`` interface share the per-position masking path.
    """
    mask = np.zeros((seq_len, len(AMINO_ACID_ORDER)))
    mask[:, :20] = 1.0
    for aa in exclude_aa:
        mask[:, AMINO_ACID_ORDER.index(aa)] = 0.0
    return mask


def resolve_aa_allowed_mask(aa_allowed_mask, exclude_aa, seq_len, default_exclude=("C",)):
    """Return ``aa_allowed_mask``, or build it from ``exclude_aa`` (``default_exclude`` if unset)."""
    if aa_allowed_mask is not None:
        return aa_allowed_mask
    if exclude_aa is None:
        exclude_aa = list(default_exclude)
    return aa_allowed_mask_from_exclude(exclude_aa, seq_len)


def apply_aa_allowed_mask(probs, aa_allowed_mask, positions):
    """Hard-zero disallowed amino acids per row of ``probs`` then renormalize.

    Disallowed amino acids stay at exactly 0 (never selectable); only the denominator is
    clamped to avoid dividing an all-zero row by zero. ``probs`` is an (n_rows, A) torch
    tensor; row ``r`` is masked by ``aa_allowed_mask`` row ``positions[r]`` (a length-1
    ``positions`` broadcasts the same row to every prob row).
    """
    rows = aa_allowed_mask[np.asarray(positions), : probs.shape[1]]
    mask = torch.as_tensor(rows, dtype=probs.dtype, device=probs.device)
    probs = probs * mask
    return probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)


def parse_position_string(position_str):
    """Parse a 1-indexed position selector into a sorted list of unique 0-indexed positions.

    Accepts comma-separated positions and inclusive ranges, e.g. '3,5-7,88-91'.
    """
    cleaned = re.sub(r"\s+", "", position_str)
    if not cleaned:
        return []
    invalid = re.findall(r"[^0-9,\-]", cleaned)
    if invalid:
        raise ValueError(f"Invalid character '{invalid[0]}' in position selector '{position_str}'.")

    positions = set()
    for token in cleaned.split(","):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Malformed range '{token}' in '{position_str}' (expected 'start-stop').")
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Range start ({start}) > end ({end}) in '{token}'.")
            if start < 1:
                raise ValueError(f"Positions must be >= 1 (1-indexed), got {start}.")
            positions.update(range(start - 1, end))
        else:
            val = int(token)
            if val < 1:
                raise ValueError(f"Positions must be >= 1 (1-indexed), got {val}.")
            positions.add(val - 1)
    return sorted(positions)


def build_aa_allowed_mask(design_regions, seq_len, seq=None, consider_wt_aa=False):
    """Build (fixed_position_mask, aa_allowed_mask) from a design-regions spec.

    Positions in no region are fixed; each region sets its positions designable and may
    restrict identities via ``allowed_aas`` or ``excluded_aas`` (not both). Regions may not
    overlap. ``aa_allowed_mask`` is (seq_len, 21); fixed positions are all-zero.
    ``consider_wt_aa`` re-adds the wild-type residue at each design position even where a
    region's ``excluded_aas`` forbade it.
    """
    if not isinstance(design_regions, dict):
        raise TypeError(f"design_regions must be a dict, got {type(design_regions).__name__}.")

    region_positions = {}
    region_aa_specs = {}
    for name, spec in design_regions.items():
        if "positions" not in spec:
            raise ValueError(f"Design region '{name}' is missing required key 'positions'.")
        parsed = parse_position_string(spec["positions"])
        out_of_range = [p + 1 for p in parsed if p >= seq_len]
        if out_of_range:
            raise ValueError(f"Design region '{name}' has positions outside 1-{seq_len}: {out_of_range}.")
        region_positions[name] = set(parsed)

        has_allowed, has_excluded = "allowed_aas" in spec, "excluded_aas" in spec
        if has_allowed and has_excluded:
            raise ValueError(f"Design region '{name}' specifies both 'allowed_aas' and 'excluded_aas'.")
        if has_allowed or has_excluded:
            key = "allowed_aas" if has_allowed else "excluded_aas"
            aa_set = set(spec[key].upper())
            invalid = aa_set - AMINO_ACID_SET
            if invalid:
                raise ValueError(f"Design region '{name}' has invalid amino acids in '{key}': {sorted(invalid)}.")
            if has_allowed and not aa_set:
                raise ValueError(f"Design region '{name}' has an empty 'allowed_aas' (no residue would be allowed).")
            region_aa_specs[name] = ("allowed" if has_allowed else "excluded", aa_set)
        else:
            region_aa_specs[name] = None

    names = list(region_positions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = region_positions[names[i]] & region_positions[names[j]]
            if overlap:
                raise ValueError(
                    f"Position(s) {sorted(p + 1 for p in overlap)} in multiple regions "
                    f"('{names[i]}', '{names[j]}'); each position may belong to only one."
                )

    fixed_position_mask = np.ones(seq_len)
    aa_allowed_mask = np.zeros((seq_len, len(AMINO_ACID_ORDER)))
    for name, pos_set in region_positions.items():
        aa_spec = region_aa_specs[name]
        row = np.zeros(len(AMINO_ACID_ORDER))
        if aa_spec is None or aa_spec[0] == "excluded":
            row[:20] = 1.0
        for aa in aa_spec[1] if aa_spec else ():
            row[AMINO_ACID_ORDER.index(aa)] = 1.0 if aa_spec[0] == "allowed" else 0.0
        positions = sorted(pos_set)
        fixed_position_mask[positions] = 0.0
        aa_allowed_mask[positions] = row
        if consider_wt_aa:
            if seq is None:
                raise ValueError("seq must be provided when consider_wt_aa is True.")
            for pos in positions:
                if seq[pos] in AMINO_ACID_ORDER:
                    aa_allowed_mask[pos, AMINO_ACID_ORDER.index(seq[pos])] = 1.0
    return fixed_position_mask, aa_allowed_mask


def parse_design_regions_arg(args):
    """Normalize ``args.design_regions`` to a dict (parsing a JSON string if needed)."""
    dr = getattr(args, "design_regions", None)
    if dr is None:
        args.design_regions = {}
        return args
    if isinstance(dr, str):
        try:
            args.design_regions = json.loads(dr)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse --design_regions as JSON: {e}") from e
    if not isinstance(args.design_regions, dict):
        raise TypeError(f"design_regions must be a dict, got {type(args.design_regions).__name__}.")
    return args


def load_config_and_merge(args, parser):
    """Merge a ``--config`` JSON file into ``args`` (JSON values override CLI, with a warning)."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: '{config_path}'")
    with open(config_path) as f:
        config = json.load(f)
    defaults = vars(parser.parse_args([]))
    for key, json_value in config.items():
        if not hasattr(args, key):
            warnings.warn(f"Config key '{key}' is not a recognized argument -- ignoring.", stacklevel=2)
            continue
        if getattr(args, key) != defaults.get(key):
            warnings.warn(
                f"CLI '--{key}' ({getattr(args, key)!r}) overridden by config value {json_value!r}.", stacklevel=2
            )
        setattr(args, key, json_value)
    return args


def resolve_protein_input(args):
    """Resolve (seq, struct) from ``args.protein_id``/``args.pdb_path`` and set ``args.protein_label``.

    ``pdb_path`` (a local file) takes priority over ``protein_id`` (fetched from RCSB).
    """
    protein_id = getattr(args, "protein_id", None)
    pdb_path = getattr(args, "pdb_path", None)
    if protein_id is None and pdb_path is None:
        raise ValueError("Either --protein_id or --pdb_path must be provided.")
    if protein_id is not None and pdb_path is not None:
        warnings.warn(
            f"Both --protein_id ('{protein_id}') and --pdb_path ('{pdb_path}') given; using --pdb_path.", stacklevel=2
        )
    if pdb_path is not None:
        args.protein_label = os.path.splitext(os.path.basename(pdb_path))[0]
        return get_protein(pdb_path=pdb_path)
    args.protein_label = protein_id
    return get_protein(pdb_code=protein_id)
