"""
split_point_policy.py — Per-model split-point boundary policies.

A policy restricts which of the N-1 dag_aligned_full boundaries may be active
in any generated mask. Disabled boundaries are forced to 0, reducing the
search space for OPT and HEU algorithms.

Policies (from configs/split_point_policies.json):
  "all"        — all N-1 boundaries enabled (no restriction)
  "paper_like" — skip fine-grained within-pair splits (Conv-ReLU merged)
  "stage"      — only at major architectural transitions (MaxPool + FC start)
  "five_points" — exactly five major design-phase boundaries per supported model
  "ten_points" — ten materializable design-phase boundaries per supported model
  "major_blocks" — paper-style major architectural block boundaries

If a model or policy is not in the JSON, defaults to "all" (no restriction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
_POLICIES_PATH = REPO / "configs" / "split_point_policies.json"

_cache: Optional[Dict] = None


def _load_policies() -> Dict:
    global _cache
    if _cache is None:
        with _policies_path().open() as f:
            _cache = json.load(f)
    return _cache


def _policies_path() -> Path:
    return _POLICIES_PATH


def get_enabled_boundaries(
    model_name: str,
    policy_name: str,
    boundary_count: int,
) -> List[int]:
    """
    Return the list of enabled boundary indices for a model+policy.

    Parameters
    ----------
    model_name     : e.g. "alexnet", "resnet18", "vgg19"
    policy_name    : "all", "paper_like", "stage", "five_points", "ten_points", or "major_blocks"
    boundary_count : N-1 for the task (safety check; all returned indices < boundary_count)

    Returns
    -------
    Sorted list of integer boundary indices. If the model/policy is not found in
    configs/split_point_policies.json, returns all indices [0..boundary_count-1].
    """
    policies = _load_policies()
    model_key = model_name.lower()

    if model_key not in policies:
        return list(range(boundary_count))

    model_policies = policies[model_key]
    policy_key = policy_name.lower()

    if policy_key not in model_policies:
        # Fallback to "all" if the named policy isn't defined
        if "all" in model_policies:
            enabled = model_policies["all"]
        else:
            return list(range(boundary_count))
    else:
        enabled = model_policies[policy_key]

    # Filter to valid range and sort
    return sorted(b for b in enabled if isinstance(b, int) and 0 <= b < boundary_count)


def apply_policy_to_mask(mask: List[int], enabled_boundaries: List[int]) -> List[int]:
    """
    Zero out any boundary in mask that is not in enabled_boundaries.

    Returns a new mask of the same length.
    """
    enabled_set = set(enabled_boundaries)
    return [b if i in enabled_set else 0 for i, b in enumerate(mask)]


def is_boundary_enabled(boundary_idx: int, enabled_boundaries: List[int]) -> bool:
    return boundary_idx in enabled_boundaries


def policy_summary(model_name: str, policy_name: str, boundary_count: int) -> str:
    enabled = get_enabled_boundaries(model_name, policy_name, boundary_count)
    return (
        f"{model_name}/{policy_name}: {len(enabled)}/{boundary_count} boundaries enabled "
        f"(indices: {enabled[:8]}{'...' if len(enabled) > 8 else ''})"
    )
