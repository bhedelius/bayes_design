"""Basic package tests.

Fast tests run anywhere. The end-to-end regression needs model weights and a
network connection (to download prot_xlnet), so it is gated behind the
BAYES_RUN_SLOW environment variable.
"""

import os

import numpy as np
import pytest

from bayes_design.model import model_dict
from bayes_design.utils import get_fixed_position_mask


def test_model_dict_has_core_models():
    for name in ("xlnet", "protein_mpnn", "bayes_design", "pssm", "trRosetta"):
        assert name in model_dict


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
        model_name="bayes_design", protein_id="6MRR", decode_order="n_to_c",
        decode_algorithm="greedy", fixed_positions=[1, 60], n_beams=16, redesign=False,
        device=0, bayes_balance_factor=0.002, temperature=1.0, n_designs=1, seed=0,
        results_dir="./results", exclude_aa=[],
    )
    out = example_design(args)
    assert out["Designed sequence"][0].endswith("GYHVNITIS")
