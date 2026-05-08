"""
test_split_policy.py — Unit tests for split_point_policy.py.
"""
import pytest


def test_get_enabled_boundaries_all_policy():
    """Policy 'all' returns all boundary indices [0..N-2]."""
    from src.integration.split_point_policy import get_enabled_boundaries
    enabled = get_enabled_boundaries("alexnet", "all", 8)
    assert enabled == list(range(8))


def test_get_enabled_boundaries_major_blocks():
    """major_blocks policy returns a subset of boundaries for known models."""
    from src.integration.split_point_policy import get_enabled_boundaries
    # AlexNet major_blocks should have fewer boundaries than 'all'
    all_b = get_enabled_boundaries("alexnet", "all", 20)
    major_b = get_enabled_boundaries("alexnet", "major_blocks", 20)
    assert len(major_b) < len(all_b)
    assert all(0 <= b < 20 for b in major_b)


def test_get_enabled_boundaries_unknown_model():
    """Unknown model falls back to all boundaries."""
    from src.integration.split_point_policy import get_enabled_boundaries
    enabled = get_enabled_boundaries("unknown_model_xyz", "major_blocks", 5)
    assert enabled == list(range(5))


def test_apply_policy_to_mask_zeros_forbidden():
    """apply_policy_to_mask zeroes out boundaries not in enabled set."""
    from src.integration.split_point_policy import apply_policy_to_mask
    mask = [1, 1, 1, 1]
    enabled = [0, 2]  # only indices 0 and 2 allowed
    result = apply_policy_to_mask(mask, enabled)
    assert result == [1, 0, 1, 0]


def test_apply_policy_to_mask_preserves_zeros():
    """apply_policy_to_mask keeps 0s that were already 0 in input."""
    from src.integration.split_point_policy import apply_policy_to_mask
    mask = [1, 0, 1, 0]
    enabled = [0, 1, 2, 3]
    result = apply_policy_to_mask(mask, enabled)
    assert result == mask


def test_policy_full_mask_respects_policy():
    """A policy-limited full-split mask has <= boundaries than unrestricted full-split."""
    from src.integration.split_point_policy import get_enabled_boundaries, apply_policy_to_mask
    N = 10
    enabled = get_enabled_boundaries("alexnet", "major_blocks", N - 1)
    policy_mask = apply_policy_to_mask([1] * (N - 1), enabled)
    unrestricted = [1] * (N - 1)
    assert sum(policy_mask) <= sum(unrestricted)
    # policy_mask must only have 1s at enabled positions
    for i, b in enumerate(policy_mask):
        if b == 1:
            assert i in enabled
