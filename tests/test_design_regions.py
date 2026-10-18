"""Tests for the multi-constraint design-region / per-position amino-acid mask layer."""

import numpy as np
import pytest

from bayes_design.utils import (
    AMINO_ACID_ORDER,
    aa_allowed_mask_from_exclude,
    build_aa_allowed_mask,
    parse_position_string,
)


def idx(aa):
    return AMINO_ACID_ORDER.index(aa)


def test_parse_position_string_mixed():
    assert parse_position_string("3,5-7,2,17,88-91") == [1, 2, 4, 5, 6, 16, 87, 88, 89, 90]
    assert parse_position_string(" 5 - 7, 3 ") == [2, 4, 5, 6]
    assert parse_position_string("") == []


@pytest.mark.parametrize("bad", ["1-", "5-3", "0", "a,2"])
def test_parse_position_string_errors(bad):
    with pytest.raises(ValueError):
        parse_position_string(bad)


def test_aa_allowed_mask_from_exclude():
    mask = aa_allowed_mask_from_exclude(["C"], 4)
    assert mask.shape == (4, len(AMINO_ACID_ORDER))
    assert mask[0, idx("C")] == 0 and mask[0, idx("X")] == 0
    assert mask[0, idx("A")] == 1
    assert mask[:, :20].sum() == 4 * 19


def test_build_aa_allowed_mask_regions():
    regions = {
        "loop": {"positions": "2-3", "excluded_aas": "C"},
        "act": {"positions": "5", "allowed_aas": "DE"},
        "free": {"positions": "7"},
    }
    fixed, mask = build_aa_allowed_mask(regions, seq_len=8, seq="ACDEFGHI")
    assert list(np.where(fixed == 0)[0]) == [1, 2, 4, 6]
    assert mask[1, idx("C")] == 0 and mask[1, idx("A")] == 1
    assert mask[4, idx("D")] == 1 and mask[4, idx("E")] == 1 and mask[4, :20].sum() == 2
    assert mask[6, :20].sum() == 20
    assert mask[0].sum() == 0 and mask[3].sum() == 0  # fixed positions are all-zero


def test_consider_wt_aa_readds_wildtype():
    _, mask = build_aa_allowed_mask({"a": {"positions": "1", "allowed_aas": "D"}}, seq_len=3, seq="ACD", consider_wt_aa=True)
    assert mask[0, idx("A")] == 1 and mask[0, idx("D")] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"x": {"positions": "1-3"}, "y": {"positions": "3-4"}},  # overlap
        {"z": {"positions": "1", "allowed_aas": "A", "excluded_aas": "C"}},  # both specs
        {"w": {"positions": "99"}},  # out of range
        {"noPos": {"allowed_aas": "A"}},  # missing positions
        {"empty": {"positions": "1", "allowed_aas": ""}},  # empty allowed set
    ],
)
def test_build_aa_allowed_mask_errors(bad):
    with pytest.raises((ValueError, KeyError)):
        build_aa_allowed_mask(bad, seq_len=8)
