"""Tests for bound-target (complex) parsing behind --design_chain/--context_chains."""

import pytest

from bayes_design.utils import get_protein_complex


def _write_minimal_pdb(path):
    """Write a 3-residue chain X then a 2-residue chain Y, each with N/CA/C/O atoms."""
    def residue(chain, n, aa3, serial):
        backbone = [("N", 0.0, 0.0), ("CA", 1.5, 0.0), ("C", 2.5, 1.0), ("O", 3.0, 1.5)]
        return [
            f"ATOM  {serial + i:>5}  {atom:<3} {aa3} {chain}{n:>4}    "
            f"{x + n:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00           {atom[0]}"
            for i, (atom, x, y) in enumerate(backbone)
        ]
    lines, serial = [], 1
    for n, aa in enumerate(["ALA", "GLY", "SER"], 1):
        lines += residue("X", n, aa, serial)
        serial += 4
    lines.append("TER")
    for n, aa in enumerate(["LEU", "VAL"], 1):
        lines += residue("Y", n, aa, serial)
        serial += 4
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")


def test_get_protein_complex_layout(tmp_path):
    pdb = tmp_path / "complex.pdb"
    _write_minimal_pdb(pdb)
    seq, struct, residue_idx, chain_encoding, design_len = get_protein_complex(str(pdb), "X", ["Y"])
    assert seq == "AGSLV"
    assert design_len == 3
    assert tuple(struct.shape) == (5, 4, 3)
    # Design chain leads with plain 0-based indices; context chain gets the 100-residue offset.
    assert residue_idx.tolist() == [0, 1, 2, 100, 101]
    assert chain_encoding.tolist() == [1, 1, 1, 2, 2]


def test_get_protein_complex_missing_chain(tmp_path):
    pdb = tmp_path / "complex.pdb"
    _write_minimal_pdb(pdb)
    with pytest.raises((ValueError, KeyError, IndexError)):
        get_protein_complex(str(pdb), "X", ["Z"])
