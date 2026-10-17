"""Basic package tests.

Fast tests run anywhere. The end-to-end regression needs model weights and a
network connection (to download prot_xlnet), so it is gated behind the
BAYES_RUN_SLOW environment variable.
"""

import os

import numpy as np
import pytest
import torch

from bayes_design.model import (
    OBJECTIVES,
    BackboneEnsembleWrapper,
    BackboneLibraryWrapper,
    CombinedModel,
    ProbabilityModel,
    model_dict,
)
from bayes_design.utils import get_fixed_position_mask, sample_backbone_noise


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
        "bayes_design_basin",
        "bayes_design_selfconsistent",
        "bayes_design_decoy",
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


class _StructSensitiveModel(ProbabilityModel):
    """Distribution whose first logit is the backbone's first coordinate, so
    different backbones give different outputs (unlike `_FixedModel`)."""

    def forward(self, seq, struct, decode_order, token_to_decode, mask_type=None, temperature=1.0):
        logits = torch.zeros(len(token_to_decode), 20)
        logits[:, 0] = struct.reshape(-1)[0]
        return torch.softmax(logits, dim=-1)


def test_sample_backbone_noise():
    struct = torch.randn(7, 4, 3)
    expanded = struct.unsqueeze(0).expand(5, 7, 4, 3)

    exact = sample_backbone_noise(struct, 5, 0.0)
    assert exact.shape == (5, 7, 4, 3)
    assert torch.allclose(exact, expanded)

    noisy = sample_backbone_noise(struct, 5, 0.3)
    assert noisy.shape == (5, 7, 4, 3)
    assert not torch.allclose(noisy, expanded)

    # Missing (NaN) coordinates are preserved.
    struct_nan = struct.clone()
    struct_nan[0, 0, 0] = float("nan")
    assert torch.isnan(sample_backbone_noise(struct_nan, 3, 0.3)[:, 0, 0, 0]).all()


def test_backbone_ensemble_reduces_to_inner():
    """n_samples=1, sigma=0 is exactly the inner model on the base backbone."""
    inner = _FixedModel(torch.softmax(torch.randn(1, 20), dim=-1))
    wrapper = BackboneEnsembleWrapper(inner, base_struct=torch.zeros(5, 4, 3), n_samples=1, sigma=0.0)
    got = wrapper(seq=["-" * 5], struct=None, decode_order=list(range(5)), token_to_decode=torch.tensor([0]))
    assert torch.allclose(got, inner._probs)


def test_backbone_ensemble_averages_over_noise():
    """The ensemble returns a valid distribution and differs from a single pass."""
    torch.manual_seed(0)
    inner = _StructSensitiveModel()
    base = torch.ones(5, 4, 3)
    wrapper = BackboneEnsembleWrapper(inner, base_struct=base, n_samples=64, sigma=1.0)
    kw = dict(seq=["-" * 5], struct=None, decode_order=list(range(5)), token_to_decode=torch.tensor([0]))

    out = wrapper(**kw)
    assert out.shape == (1, 20)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-5)
    single = inner(seq=["-" * 5], struct=base, decode_order=list(range(5)), token_to_decode=torch.tensor([0]))
    assert not torch.allclose(out, single)


def test_backbone_library_averages_cropped_backbones():
    """The library wrapper crops each backbone to the query length and averages,
    skipping entries shorter than the query."""
    inner = _StructSensitiveModel()
    library = [
        torch.full((5, 4, 3), 2.0),  # usable, cropped to 3; first coord 2.0
        torch.full((8, 4, 3), 4.0),  # usable, cropped to 3; first coord 4.0
        torch.full((2, 4, 3), 9.0),  # too short, skipped
    ]
    wrapper = BackboneLibraryWrapper(inner, library)
    kw = dict(seq=["-" * 3], struct=None, decode_order=list(range(3)), token_to_decode=torch.tensor([0]))

    out = wrapper(**kw)
    expected = (inner(struct=torch.full((3, 4, 3), 2.0), **kw) + inner(struct=torch.full((3, 4, 3), 4.0), **kw)) / 2
    assert torch.allclose(out, expected)


def test_backbone_library_respects_n_samples():
    """n_samples caps how many usable backbones are averaged."""
    inner = _FixedModel(torch.softmax(torch.randn(1, 20), dim=-1))
    library = [torch.zeros(4, 4, 3) for _ in range(5)]
    wrapper = BackboneLibraryWrapper(inner, library, n_samples=2)
    got = wrapper(seq=["-" * 4], struct=None, decode_order=list(range(4)), token_to_decode=torch.tensor([0]))
    assert torch.allclose(got, inner._probs)


def test_backbone_library_raises_when_all_too_short():
    inner = _FixedModel(torch.softmax(torch.randn(1, 20), dim=-1))
    wrapper = BackboneLibraryWrapper(inner, [torch.zeros(2, 4, 3)])
    with pytest.raises(ValueError, match="at least 4 residues"):
        wrapper(seq=["-" * 4], struct=None, decode_order=list(range(4)), token_to_decode=torch.tensor([0]))


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
        decoy_id=None,
        ligand_file=None,
        ligand_file_b=None,
        ball_mask=False,
        contrast_weight=1.0,
        n_ensemble=8,
        ensemble_sigma=0.1,
        decoy_library=None,
    )
    out = example_design(args)
    assert out["Designed sequence"][0].endswith("GYHVNITIS")
