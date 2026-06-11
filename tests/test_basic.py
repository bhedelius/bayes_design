"""Basic package tests.

Fast tests run anywhere. The end-to-end regression needs model weights and a
network connection (to download prot_xlnet), so it is gated behind the
BAYES_RUN_SLOW environment variable.
"""

import os

import numpy as np
import pytest
import torch

from bayes_design.model import OBJECTIVES, CombinedModel, ProbabilityModel, model_dict
from bayes_design.utils import get_fixed_position_mask


def test_model_dict_has_core_models():
    for name in ("xlnet", "protein_mpnn", "bayes_design", "pssm", "trRosetta"):
        assert name in model_dict


def test_objectives_registered():
    for name in (
        "bayes_design",
        "bayes_design_ligand",
        "bayes_design_ligand_specificity",
        "bayes_design_soluble",
        "bayes_design_esm_if1",
        "bayes_design_ensemble",
    ):
        assert name in OBJECTIVES and name in model_dict


class _FixedModel(ProbabilityModel):
    """A ProbabilityModel that returns a preset (N x 20) distribution."""

    def __init__(self, probs):
        super().__init__()
        self._probs = probs

    def forward(self, seq, struct, decode_order, token_to_decode, mask_type=None, temperature=1.0):
        return self._probs


def test_combined_model_two_term_matches_ratio():
    """The 2-term (+1, -1) CombinedModel reproduces the old (p_num+b)/(p_den+b) ratio."""
    torch.manual_seed(0)
    p_num = torch.softmax(torch.randn(1, 20), dim=-1)
    p_den = torch.softmax(torch.randn(1, 20), dim=-1)
    b = 0.002

    combined = CombinedModel([(_FixedModel(p_num), 1.0), (_FixedModel(p_den), -1.0)], balance=b)
    got = combined(seq=["A"], struct=None, decode_order=[0], token_to_decode=torch.tensor([0]))

    ratio = (p_num + b) / (p_den + b)
    expected = ratio / ratio.sum(dim=-1, keepdim=True)
    assert torch.allclose(got, expected, atol=1e-5)


def test_kabsch_alignment_recovers_rigid_transform():
    """Kabsch alignment undoes an arbitrary rotation+translation."""
    from bayes_design.experiments.scrmsd import kabsch_alignment

    rng = np.random.default_rng(0)
    coords1 = rng.random((10, 3))
    alpha, beta, gamma = np.pi / 4, np.pi / 3, np.pi / 6
    rot = np.array(
        [
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
                np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
            ],
            [
                np.sin(alpha) * np.cos(beta),
                np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
                np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),
            ],
            [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)],
        ]
    )
    coords2 = np.dot(rot, coords1.T).T + rng.random(3)
    aligned = kabsch_alignment(coords1, coords2)
    assert not np.allclose(coords1, coords2)
    assert np.allclose(coords1, aligned)


def test_calculate_identity():
    from bayes_design.experiments.scrmsd import calculate_identity

    assert calculate_identity("ACDE", "ACDE") == 100.0
    assert calculate_identity("ACDE", "ACDX") == 75.0


def test_fixed_position_mask():
    # Fix residues 3-5 and 8-8 (1-indexed, inclusive) in a length-10 sequence.
    mask = get_fixed_position_mask(fixed_position_list=[3, 5, 8, 8], seq_len=10)
    assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
    assert isinstance(mask, np.ndarray)


@pytest.mark.skipif(not os.environ.get("BAYES_RUN_SLOW"), reason="needs weights + network")
def test_bayes_design_regression():
    """bayes_design on 6MRR (greedy, fix 1-60) reproduces the known design tail."""
    import argparse

    from bayes_design.cli import example_design

    args = argparse.Namespace(
        model_name="bayes_design",
        protein_id="6MRR",
        decode_order="n_to_c",
        decode_algorithm="greedy",
        fixed_positions=[1, 60],
        n_beams=16,
        redesign=False,
        device=0,
        bayes_balance_factor=0.002,
        temperature=1.0,
        n_designs=1,
        seed=0,
        results_dir="./results",
        exclude_aa=[],
    )
    out = example_design(args)
    assert out["Designed sequence"][0].endswith("GYHVNITIS")
